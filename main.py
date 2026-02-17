import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from app.handler import router

import os
from dotenv import load_dotenv


load_dotenv()

TOKEN = os.getenv('BOT_TOKEN')

dp = Dispatcher()


async def main() -> None:
    # All handlers should be attached to the Router (or Dispatcher)
    dp = Dispatcher()   
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())