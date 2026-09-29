import asyncio
import base64

import json

from pathlib import Path



from fastapi import WebSocket, WebSocketDisconnect

from google.genai import types



from app.realtime.gemini_live import GeminiLiveSession

from app.conversation.session import ConversationSession

from app.context.builder import build_session_context

from app.context.semantic_context_builder import build_semantic_context

from app.rag_client import generate_report

from app.auth.firebase_auth import verify_firebase_token

from app.services.session_store import (

    create_session,

    mark_session_processing,

    mark_session_completed,

    mark_session_failed,

    register_artifact,

)

from app.services.storage import upload_artifact





# ================================================================

# FIREBASE AUTHENTICATION

# ================================================================



async def authenticate_websocket(

    websocket: WebSocket,

) -> dict:

    """

    Wait for the first WebSocket message and require it to contain

    a Firebase ID token.



    Expected message:



    {

        "type": "auth",

        "token": "\<firebase-id-token>"

    }



    Returns the decoded Firebase token claims.

    """



    try:

        message = await websocket.receive()



        # ----------------------------------------------------------

        # BROWSER DISCONNECTED BEFORE AUTHENTICATION

        # ----------------------------------------------------------



        if message.get("type") == "websocket.disconnect":

            raise WebSocketDisconnect()



        # ----------------------------------------------------------

        # AUTHENTICATION MUST BE A TEXT MESSAGE

        # ----------------------------------------------------------



        if message.get("text") is None:

            raise ValueError(

                "Authentication message must be JSON text."

            )



        try:

            data = json.loads(message["text"])

        except json.JSONDecodeError:

            raise ValueError(

                "Invalid authentication message."

            )



        # ----------------------------------------------------------

        # CHECK MESSAGE TYPE

        # ----------------------------------------------------------



        if not isinstance(data, dict):

            raise ValueError(

                "Invalid authentication message."

            )



        if data.get("type") != "auth":

            raise ValueError(

                "Authentication is required before starting a session."

            )



        # ----------------------------------------------------------

        # GET FIREBASE ID TOKEN

        # ----------------------------------------------------------



        id_token = data.get("token")



        if not isinstance(id_token, str) or not id_token.strip():

            raise ValueError(

                "Firebase authentication token is missing."

            )



        # ----------------------------------------------------------

        # VERIFY TOKEN WITH FIREBASE ADMIN

        # ----------------------------------------------------------



        decoded_token = verify_firebase_token(

            id_token.strip()

        )



        firebase_uid = decoded_token.get("uid")



        if not firebase_uid:

            raise ValueError(

                "Firebase token does not contain a user ID."

            )



        print(

            f"[AUTH] Firebase user verified: "

            f"{firebase_uid}"

        )



        # ----------------------------------------------------------

        # TELL FRONTEND AUTHENTICATION SUCCEEDED

        # ----------------------------------------------------------



        await websocket.send_json({

            "type": "auth_ok",

        })



        return decoded_token



    except WebSocketDisconnect:

        raise



    except Exception as error:

        print(

            "[AUTH] Authentication failed:",

            repr(error),

        )



        try:

            await websocket.send_json({

                "type": "error",

                "message": "Authentication failed.",

            })

        except Exception:

            pass



        try:

            await websocket.close(code=1008)

        except Exception:

            pass



        raise





# ================================================================

# BROWSER -> BACKEND

# ================================================================



async def receive_from_browser(

    websocket: WebSocket,

    gemini_session,

    conversation: ConversationSession,

    ending_event: asyncio.Event,

):

    try:

        while True:

            message = await websocket.receive()



            # ------------------------------------------------------

            # BROWSER DISCONNECTED

            # ------------------------------------------------------



            if message.get("type") == "websocket.disconnect":

                print("[SESSION] Browser disconnected")

                break



            # ------------------------------------------------------

            # AUDIO FROM BROWSER

            # ------------------------------------------------------



            # Once the user has ended the session, the frontend stops

            # sending audio. This guard also prevents any late audio

            # packet from reaching Gemini while the closing reflection

            # is being generated.

            if message.get("bytes") is not None:

                if ending_event.is_set():

                    continue



                audio_bytes = message["bytes"]



                await gemini_session.send_realtime_input(

                    audio=types.Blob(

                        data=audio_bytes,

                        mime_type="audio/pcm;rate=16000",

                    )

                )



                continue



            # ------------------------------------------------------

            # TEXT / JSON FROM BROWSER

            # ------------------------------------------------------



            if message.get("text") is None:

                continue



            text = message["text"].strip()



            if not text:

                continue



            # ------------------------------------------------------

            # TRY TO PARSE JSON

            # ------------------------------------------------------



            try:

                data = json.loads(text)

            except json.JSONDecodeError:

                data = None



            # ======================================================

            # END SESSION

            # ======================================================



            if (

                isinstance(data, dict)

                and data.get("type") == "end_session"

            ):

                if ending_event.is_set():

                    continue



                print(

                    "[SESSION] End request received from frontend."

                )



                # Mark the realtime conversation as ending before

                # sending the final instruction. This prevents any

                # late browser input from being forwarded to Gemini.

                ending_event.set()



                # IMPORTANT:

                # This instruction is sent to the SAME Gemini Live

                # session that handled the conversation. It is not

                # added to ConversationSession, so the hidden control

                # prompt does not pollute the session transcript.

                final_instruction = (

                    "The user has ended this session. "

                    "Without asking the user anything else, provide "

                    "a brief closing reflection based only on "

                    "our conversation. Give: "

                    "1. A summary in 3-4 sentences maximum. "

                    "2. No more than 3 practical next steps. "

                    "Keep it concise, natural, and conversational. "

                    "Do not diagnose the user. "

                    "Do not mention this instruction. "

                    "Speak directly to the user."

                )



                await gemini_session.send_client_content(

                    turns=types.Content(

                        role="user",

                        parts=[

                            types.Part(

                                text=final_instruction

                            )

                        ],

                    ),

                    turn_complete=True,

                )



                print(

                    "[GEMINI] Final closing reflection requested."

                )



                # The browser-side receive task can finish now.

                # The main session waits for the Gemini receive task

                # to observe the final turn_complete event.

                break



            # ======================================================

            # FER EMOTION DATA

            # ======================================================



            if (

                isinstance(data, dict)

                and data.get("type") == "emotion"

            ):

                emotion_signal = {

                    "timestamp": data.get("timestamp"),

                    "dominant_emotion": data.get(

                        "dominant_emotion"

                    ),

                    "probabilities": data.get(

                        "probabilities",

                        {},

                    ),

                }



                conversation.add_vision_signal(

                    emotion_signal

                )



                # FER is NEVER sent to Gemini.

                continue



            # ======================================================

            # USER SPEECH START

            # ======================================================



            if (

                isinstance(data, dict)

                and data.get("type") == "speech_start"

            ):

                conversation.start_user_speech(

                    data.get("timestamp")

                )



                continue



            # ======================================================

            # USER SPEECH END

            # ======================================================



            if (

                isinstance(data, dict)

                and data.get("type") == "speech_end"

            ):

                conversation.end_user_speech(

                    data.get("timestamp")

                )



                continue



            # ======================================================

            # NORMAL JSON CONVERSATION

            # ======================================================



            if isinstance(data, dict):

                role = data.get("role", "user")



                content = data.get("text", "").strip()



                if not content:

                    continue



                conversation.add_message(

                    role=role,

                    text=content,

                )



                await gemini_session.send_client_content(

                    turns=types.Content(

                        role="user",

                        parts=[

                            types.Part(

                                text=content

                            )

                        ],

                    ),

                    turn_complete=True,

                )



                continue



            # ======================================================

            # PLAIN TEXT FALLBACK

            # ======================================================



            conversation.add_message(

                role="user",

                text=text,

            )



            await gemini_session.send_client_content(

                turns=types.Content(

                    role="user",

                    parts=[

                        types.Part(

                            text=text

                        )

                    ],

                ),

                turn_complete=True,

            )



    except WebSocketDisconnect:

        print("[SESSION] Browser WebSocket disconnected")



    except asyncio.CancelledError:

        raise



    except Exception as error:

        print(

            "[ERROR] Browser receive:",

            repr(error),

        )

        raise





# ================================================================

# GEMINI -> BACKEND -> BROWSER

# ================================================================



async def receive_from_gemini(

    websocket: WebSocket,

    gemini_session,

    conversation: ConversationSession,

    ending_event: asyncio.Event,

):

    try:

        while True:



            async for response in gemini_session.receive():



                server_content = response.server_content



                if server_content is None:

                    continue



                # ==================================================

                # USER TRANSCRIPTION

                # ==================================================



                input_transcription = (

                    server_content.input_transcription

                )



                if input_transcription is not None:



                    text = input_transcription.text



                    if text:



                        await websocket.send_json({

                            "type": "transcription",

                            "role": "user",

                            "text": text,

                            "finished": input_transcription.finished,

                        })



                        conversation.add_user_transcription(

                            text=text

                        )



                # ==================================================

                # USER INTERIM TRANSCRIPTION

                # ==================================================



                interim_transcription = (

                    server_content.interim_input_transcription

                )



                if interim_transcription is not None:



                    text = interim_transcription.text



                    if text:



                        await websocket.send_json({

                            "type": "transcription_interim",

                            "role": "user",

                            "text": text,

                        })



                # ==================================================

                # AI TRANSCRIPTION

                # ==================================================



                output_transcription = (

                    server_content.output_transcription

                )



                if output_transcription is not None:



                    text = output_transcription.text



                    if text:



                        await websocket.send_json({

                            "type": "transcription",

                            "role": "assistant",

                            "text": text,

                            "finished": output_transcription.finished,

                        })



                        # The automatic closing reflection is a

                        # control response, not part of the raw

                        # conversation transcript used by RAG.

                        if not ending_event.is_set():

                            conversation.add_assistant_transcription(

                                text=text

                            )



                # ==================================================

                # TURN COMPLETE

                # ==================================================



                if server_content.turn_complete:

                    # The final reflection is intentionally excluded

                    # from ConversationSession. It exists only to

                    # improve the user's end-of-session experience.

                    if not ending_event.is_set():

                        conversation.complete_turn()

                    else:

                        print(

                            "[GEMINI] Final closing reflection "

                            "completed."

                        )

                        return



                # ==================================================

                # GEMINI AUDIO

                # ==================================================



                if response.data:



                    try:

                        await websocket.send_bytes(

                            response.data

                        )



                    except Exception:

                        return



                # ==================================================

                # IMPORTANT:

                #

                # DO NOT USE response.text HERE.

                # ==================================================



    except asyncio.CancelledError:

        raise



    except Exception as error:

        print(

            "[ERROR] Gemini receive:",

            repr(error),

        )

        raise





# ================================================================

# MAIN REALTIME SESSION

# ================================================================



async def realtime_conversation(

    websocket: WebSocket,

):

    await websocket.accept()



    browser_task = None

    gemini_task = None



    conversation = None

    gemini = None



    firebase_uid = None



    database_session_created = False

    processing_marked = False



    # Set by receive_from_browser when the frontend explicitly

    # requests the automatic closing reflection.

    ending_event = asyncio.Event()



    try:



        # ==========================================================

        # AUTHENTICATE USER BEFORE CREATING SESSION

        # ==========================================================



        decoded_token = await authenticate_websocket(

            websocket

        )



        firebase_uid = decoded_token["uid"]



        print(

            f"[AUTH] Starting session for Firebase user: "

            f"{firebase_uid}"

        )



        # ==========================================================

        # CREATE CONVERSATION SESSION

        # ==========================================================



        conversation = ConversationSession()



        print()

        print(

            f"[SESSION] Started: "

            f"{conversation.session_id}"

        )



        # ==========================================================

        # CREATE SUPABASE SESSION

        # ==========================================================



        database_session = create_session(

            firebase_uid=firebase_uid,

            session_id=conversation.session_id,

        )



        database_session_created = True



        # ==========================================================

        # SEND AUTHORITATIVE SESSION ID TO FRONTEND

        # ==========================================================



        await websocket.send_json({

            "type": "session_started",

            "session_id": conversation.session_id,

        })



        print(

            f"[SESSION] Session ID sent to frontend: "

            f"{conversation.session_id}"

        )



        print(

            f"[DB] Session created: "

            f"{database_session.get('session_id')}"

        )



        print(

            f"[DB] Status: "

            f"{database_session.get('status')}"

        )



        # ==========================================================

        # CONNECT GEMINI

        # ==========================================================



        gemini = GeminiLiveSession()



        session = await gemini.connect()



        print("[GEMINI] Connected")



        # ==========================================================

        # START INDEPENDENT TASKS

        # ==========================================================



        browser_task = asyncio.create_task(

            receive_from_browser(

                websocket,

                session,

                conversation,

                ending_event,

            )

        )



        gemini_task = asyncio.create_task(

            receive_from_gemini(

                websocket,

                session,

                conversation,

                ending_event,

            )

        )



        # ==========================================================

        # WAIT FOR THE REALTIME SESSION TO FINISH

        # ==========================================================



        done, pending = await asyncio.wait(

            [

                browser_task,

                gemini_task,

            ],

            return_when=asyncio.FIRST_COMPLETED,

        )



        # ----------------------------------------------------------

        # NORMAL DISCONNECT / UNEXPECTED END

        # ----------------------------------------------------------



        if not ending_event.is_set():



            for task in pending:

                task.cancel()



            if pending:

                await asyncio.gather(

                    *pending,

                    return_exceptions=True,

                )



        # ----------------------------------------------------------

        # USER REQUESTED END SESSION

        # ----------------------------------------------------------



        else:

            print(

                "[SESSION] Waiting for Gemini closing reflection..."

            )



            # receive_from_browser has finished after sending the

            # hidden final instruction. Keep receive_from_gemini

            # alive so the final audio and transcription can reach

            # the frontend.

            if not gemini_task.done():

                await gemini_task



            # The browser task should already be finished, but wait

            # for it explicitly so cleanup is deterministic.

            if not browser_task.done():

                await browser_task



            print(

                "[SESSION] Gemini closing reflection finished."

            )



        # ==========================================================

        # CHECK COMPLETED TASK ERRORS

        # ==========================================================



        for task in [

            browser_task,

            gemini_task,

        ]:



            if task is None or not task.done():

                continue



            try:

                exception = task.exception()



                if exception:

                    print(

                        "[ERROR] Realtime task:",

                        repr(exception),

                    )



            except asyncio.CancelledError:

                pass



    except WebSocketDisconnect:

        print("[SESSION] Browser WebSocket disconnected")



    except Exception as error:



        print(

            "[ERROR] Realtime WebSocket:",

            repr(error),

        )



        try:

            await websocket.send_json({

                "type": "error",

                "message": str(error),

            })



        except Exception:

            pass



    finally:



        # ==========================================================

        # NO SESSION WAS CREATED

        # ==========================================================



        if conversation is None:



            if gemini is not None:

                try:

                    await gemini.close()

                except Exception:

                    pass



            return



        # ==========================================================

        # FINISH SESSION

        # ==========================================================



        conversation.end_session()



        # ==========================================================

        # MARK DATABASE SESSION AS PROCESSING

        # ==========================================================



        if database_session_created:



            try:



                mark_session_processing(

                    session_id=conversation.session_id

                )



                processing_marked = True



                print(

                    "[DB] Session status: processing"

                )



            except Exception as error:



                print(

                    "[DB] Failed to mark session as processing:",

                    repr(error),

                )



        # ==========================================================

        # POST-SESSION PROCESSING

        # ==========================================================



        processing_succeeded = False



        try:



            # ======================================================

            # BUILD EXISTING SESSION CONTEXT

            # ======================================================



            session_data = conversation.get_session_data()



            session_context = build_session_context(

                session_data

            )



            session_directory = (

                Path(__file__).resolve().parents[2]

                / "sessions"

                / conversation.session_id

            )



            session_directory.mkdir(

                parents=True,

                exist_ok=True,

            )



            # ======================================================

            # SAVE EXISTING SESSION CONTEXT

            # ======================================================



            context_file = (

                session_directory

                / "session_context.json"

            )



            with context_file.open(

                "w",

                encoding="utf-8",

            ) as file:

                json.dump(

                    session_context,

                    file,

                    indent=2,

                    ensure_ascii=False,

                )



            # ======================================================

            # BUILD SEMANTIC CONTEXT

            # ======================================================



            semantic_context = build_semantic_context(

                session_context

            )



            # ======================================================

            # SAVE SEMANTIC CONTEXT

            # ======================================================



            semantic_context_file = (

                session_directory

                / "semantic_context.json"

            )



            with semantic_context_file.open(

                "w",

                encoding="utf-8",

            ) as file:

                json.dump(

                    semantic_context,

                    file,

                    indent=2,

                    ensure_ascii=False,

                )



            # ======================================================

            # GENERATE RAG REPORT

            # ======================================================



            rag_result = await generate_report(

                semantic_context

            )



            print()

            print(

                "=" * 60

            )

            print(

                "RAG REPORT GENERATED"

            )

            print(

                "=" * 60

            )



            print(

                f"[RAG] Session ID: "

                f"{rag_result.get('session_id')}"

            )



            print(

                f"[RAG] Report: "

                f"{rag_result.get('report_filename')}"

            )



            assessment = (

                rag_result.get(

                    "assessment",

                    {},

                )

            )



            print(

                f"[RAG] Assessment: "

                f"{assessment}"

            )



            # ======================================================

            # UPLOAD SESSION ARTIFACTS TO SUPABASE STORAGE

            # ======================================================



            print()

            print(

                "=" * 60

            )

            print(

                "UPLOADING SESSION ARTIFACTS"

            )

            print(

                "=" * 60

            )



            # ------------------------------------------------------

            # SESSION CONTEXT JSON

            # ------------------------------------------------------



            session_context_storage_path = upload_artifact(

                firebase_uid=firebase_uid,

                session_id=conversation.session_id,

                artifact_type="session_context",

                file_bytes=context_file.read_bytes(),

                content_type="application/json",

            )



            register_artifact(

                session_id=conversation.session_id,

                artifact_type="session_context",

                storage_path=session_context_storage_path,

            )



            print(

                "[STORAGE] Uploaded session_context.json"

            )



            # ------------------------------------------------------

            # SEMANTIC CONTEXT JSON

            # ------------------------------------------------------



            semantic_context_storage_path = upload_artifact(

                firebase_uid=firebase_uid,

                session_id=conversation.session_id,

                artifact_type="semantic_context",

                file_bytes=semantic_context_file.read_bytes(),

                content_type="application/json",

            )



            register_artifact(

                session_id=conversation.session_id,

                artifact_type="semantic_context",

                storage_path=semantic_context_storage_path,

            )



            print(

                "[STORAGE] Uploaded semantic_context.json"

            )



            # ------------------------------------------------------

            # RAG PDF

            # ------------------------------------------------------



            report_base64 = rag_result.get(
                "report_base64"
            )

            if not report_base64:
                raise RuntimeError(
                    "RAG result does not contain report_base64."
                )

            try:
                report_bytes = base64.b64decode(
                    report_base64,
                    validate=True,
                )
            except Exception as error:
                raise RuntimeError(
                    "RAG result contains invalid report_base64."
                ) from error

            if not report_bytes:
                raise RuntimeError(
                    "Decoded RAG PDF is empty."
                )

            report_storage_path = upload_artifact(
                firebase_uid=firebase_uid,
                session_id=conversation.session_id,
                artifact_type="report_pdf",
                file_bytes=report_bytes,
                content_type="application/pdf",
            )



            register_artifact(

                session_id=conversation.session_id,

                artifact_type="report_pdf",

                storage_path=report_storage_path,

            )



            print(

                "[STORAGE] Uploaded report.pdf"

            )



            # ======================================================

            # MARK DATABASE SESSION COMPLETED

            # ======================================================



            if database_session_created:



                try:



                    completed_session = (

                        mark_session_completed(

                            session_id=conversation.session_id,

                            rag_result=rag_result,

                        )

                    )



                    print(

                        "[DB] Session status: completed"

                    )



                    print(

                        "[DB] Summary saved"

                    )



                except Exception as error:



                    print(

                        "[DB] Failed to mark session as completed:",

                        repr(error),

                    )



                    raise



            # ======================================================

            # CLEAN LOCAL ARTIFACTS

            #

            # Only reached after:

            #

            # 1. session_context uploaded

            # 2. semantic_context uploaded

            # 3. report uploaded

            # 4. all artifact metadata registered

            # 5. database session marked completed

            #

            # If anything above fails, this cleanup is skipped.

            # ======================================================



            try:



                if context_file.exists():

                    context_file.unlink()



                if semantic_context_file.exists():

                    semantic_context_file.unlink()



                if session_directory.exists():

                    try:

                        session_directory.rmdir()

                    except OSError:

                        pass



                print(

                    "[CLEANUP] Local session JSON files removed"

                )



                # The RAG backend owns the generated PDF on its
                # own filesystem. The Conversation backend receives
                # the PDF as base64 data and uploads it directly to
                # Supabase, so there is no local RAG PDF to remove.
            except Exception as cleanup_error:



                # Cleanup failure should NOT turn an otherwise

                # successful session into a failed session.

                print(

                    "[CLEANUP] Local artifact cleanup failed:",

                    repr(cleanup_error),

                )



            processing_succeeded = True



        except Exception as error:



            print()

            print(

                "=" * 60

            )

            print(

                "POST-SESSION PROCESSING FAILED"

            )

            print(

                "=" * 60

            )



            print(

                f"[PROCESSING] Error: "

                f"{repr(error)}"

            )



            # ======================================================

            # MARK DATABASE SESSION FAILED

            # ======================================================



            if database_session_created:



                try:



                    mark_session_failed(

                        session_id=conversation.session_id

                    )



                    print(

                        "[DB] Session status: failed"

                    )



                except Exception as database_error:



                    print(

                        "[DB] Failed to mark session as failed:",

                        repr(database_error),

                    )



        # ==========================================================

        # CLEAN TERMINAL SUMMARY

        # ==========================================================



        print(

            f"[AUTH] Firebase UID: "

            f"{firebase_uid}"

        )



        print(

            "[STORAGE] session_context.json uploaded"

        )



        print(

            "[STORAGE] semantic_context.json uploaded"

        )



        print(

            "[STORAGE] report.pdf uploaded"

        )



        print(

            f"[STATS] "

            f"{len(conversation.messages)} messages | "

            f"{len(conversation.vision_signals)} vision signals | "

            f"{len(conversation.user_speech_intervals)} speech intervals"

        )



        if processing_succeeded:



            print(

                "[STATUS] Session processing completed successfully"

            )



        elif processing_marked:



            print(

                "[STATUS] Session processing failed"

            )



        # ==========================================================

        # CLEANUP TASKS

        # ==========================================================



        tasks_to_wait = []



        for task in [

            browser_task,

            gemini_task,

        ]:



            if task is not None:



                if not task.done():

                    task.cancel()



                tasks_to_wait.append(task)



        if tasks_to_wait:



            await asyncio.gather(

                *tasks_to_wait,

                return_exceptions=True,

            )



        # ==========================================================

        # CLOSE GEMINI

        # ==========================================================



        if gemini is not None:



            try:

                await gemini.close()

            except Exception as error:

                print(

                    "[GEMINI] Close error:",

                    repr(error),

                )



        print(

            f"[SESSION] Ended: "

            f"{conversation.session_id}"

        )
