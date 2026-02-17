from aiogram import F, Router, html
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from parse_new_url import main
router = Router()

@router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")


@router.message(Command(commands="help"))
async def command_help_handler(message: Message) -> None:
    """
    This handler receives messages with `/help` command
    """
    await message.answer("This is 'help' section. The purpose of this bot is to check URLs for phishing threats. Just send me a URL, and I'll analyze it for you! Remember, these checks are automated and may not be 100% accurate. If you have any concerns about a URL, it's best to avoid clicking on it and consult a cybersecurity professional. We are not responsible for any actions you take based on the analysis provided by this bot. Stay safe online!")


@router.message()
async def url_handler(message: Message) -> None:
    """
    Handler will check if the message is a URL and tell if it's phishing or not.
    """
    try:
        await message.answer("Analyzing the URL you provided...")
        url = message.text
        await message.answer(main(url))
        
    except TypeError:
        # But not all the types is supported to be copied so need to handle it
        await message.answer("Sorry, I accept only urls for now.")