import asyncio

from app.realtime.gemini_live import GeminiLiveSession


async def main():

    print("Starting Gemini Live test...")

    gemini = GeminiLiveSession()

    try:

        session = await gemini.connect()

        print()
        print("=" * 50)
        print("SUCCESS!")
        print("Gemini Live connection is working.")
        print("=" * 50)
        print()

        await asyncio.sleep(5)

    except Exception as error:

        print()
        print("=" * 50)
        print("GEMINI CONNECTION FAILED")
        print("=" * 50)
        print(error)
        print()

    finally:

        await gemini.close()


if __name__ == "__main__":

    asyncio.run(main())