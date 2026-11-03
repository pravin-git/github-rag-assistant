import asyncio

from app.github_client import connect_github


async def main():
    adapter, tools = await connect_github()

    print(f"\nTotal tools discovered: {len(tools)}")

    await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())