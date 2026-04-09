import sys
import logging
from typing import Protocol

from src.config import (
    setup_logging,
    get_app_config,
    get_drive_config,
    get_image_editor_config,
    get_caption_config,
    get_telegram_config,
    get_tracking_config,
    get_value,
    load_config
)
from src.agents.image_editor_agent import ImageEditorAgent
from src.agents.caption_agent import CaptionAgent
from src.agents import instagram_workflow as workflow
from src.clients import DriveClient, TelegramNotifier
from src.utils.image_tracker import ImageTracker
from src.scheduler.polling_scheduler import PollingScheduler


class ImageEditor(Protocol):
    def edit_image(self, image_content: bytes) -> bytes: ...


class CaptionGenerator(Protocol):
    def generate_caption(self, image_content: bytes) -> str: ...


class DriveClientInterface(Protocol):
    def list_images(self, folder_id: str) -> list[dict]: ...
    def download_image(self, file_id: str) -> bytes: ...
    def upload_image(self, folder_id: str, file_name: str, content: bytes, mime_type: str) -> str: ...
    input_folder_id: str
    output_folder_id: str


class TelegramClient(Protocol):
    def send_image_with_caption(self, image_content: bytes, caption: str) -> bool: ...
    def send_error_notification(self, error_message: str, context: str = "") -> bool: ...


class ImageTrackerInterface(Protocol):
    def is_processed(self, file_id: str) -> bool: ...
    def mark_processed(self, file_id: str) -> None: ...


def create_drive_client(
    service_account_json: str,
    credentials_path: str | None,
    input_folder: str,
    output_folder: str
) -> DriveClient:
    """Factory function to create DriveClient."""
    client = DriveClient(
        service_account_json=service_account_json,
        credentials_path=credentials_path
    )
    client.input_folder_id = input_folder
    client.output_folder_id = output_folder
    return client


def create_telegram_notifier(
    bot_token: str,
    chat_id: str,
    enabled: bool
) -> TelegramNotifier:
    """Factory function to create TelegramNotifier."""
    return TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        enabled=enabled
    )


def create_image_editor(config: dict) -> ImageEditorAgent:
    """Factory function to create ImageEditorAgent."""
    return ImageEditorAgent(
        model=get_value('IMAGE_MODEL', config.get('model', 'qwen/qwen2-vl-7b-instruct'), 'IMAGE_MODEL'),
        prompt_file=config.get('prompt_file', 'config/prompts/image_editor.md'),
        max_retries=config.get('max_retries', 3),
        retry_delay=config.get('retry_delay_seconds', 5)
    )


def create_caption_agent(config: dict) -> CaptionAgent:
    """Factory function to create CaptionAgent."""
    return CaptionAgent(
        model=get_value('CAPTION_MODEL', config.get('model', 'groq/llama-3.3-70b-versatile'), 'CAPTION_MODEL'),
        prompt_file=config.get('prompt_file', 'config/prompts/caption.md'),
        max_retries=config.get('max_retries', 3),
        retry_delay=config.get('retry_delay_seconds', 2),
        fallback_caption=config.get('fallback_caption', '✨ New photo posted')
    )


def create_image_tracker(config: dict) -> ImageTracker:
    """Factory function to create ImageTracker."""
    return ImageTracker(config.get('file', 'processed_images.json'))


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Instagram Agent...")
    
    load_config()
    
    app_config = get_app_config()
    drive_config = get_drive_config()
    image_config = get_image_editor_config()
    caption_config = get_caption_config()
    telegram_config = get_telegram_config()
    tracking_config = get_tracking_config()
    
    input_folder = get_value('GOOGLE_DRIVE_INPUT_FOLDER_ID', drive_config.get('input_folder_id', ''), 'GOOGLE_DRIVE_INPUT_FOLDER_ID')
    output_folder = get_value('GOOGLE_DRIVE_OUTPUT_FOLDER_ID', drive_config.get('output_folder_id', ''), 'GOOGLE_DRIVE_OUTPUT_FOLDER_ID')
    
    if not input_folder or not output_folder:
        logger.error("GOOGLE_DRIVE_INPUT_FOLDER_ID and GOOGLE_DRIVE_OUTPUT_FOLDER_ID are required")
        sys.exit(1)
    
    service_account_json = get_value('GOOGLE_SERVICE_ACCOUNT_JSON', '', 'GOOGLE_SERVICE_ACCOUNT_JSON')
    credentials_path = None
    if not service_account_json:
        credentials_path = get_value('GOOGLE_CREDENTIALS_PATH', 'credentials.json', 'GOOGLE_CREDENTIALS_PATH')
    
    drive_client = create_drive_client(
        service_account_json,
        credentials_path,
        input_folder,
        output_folder
    )
    
    telegram_token = get_value('TELEGRAM_BOT_TOKEN', '', 'TELEGRAM_BOT_TOKEN')
    telegram_chat_id = get_value('TELEGRAM_CHAT_ID', telegram_config.get('chat_id', ''), 'TELEGRAM_CHAT_ID')
    telegram_enabled = telegram_config.get('enabled', True) and telegram_token and telegram_chat_id
    
    telegram_notifier = create_telegram_notifier(telegram_token, telegram_chat_id, telegram_enabled)
    
    image_editor = create_image_editor(image_config)
    caption_agent = create_caption_agent(caption_config)
    tracker = create_image_tracker(tracking_config)
    
    def poll():
        logger.info("Running poll cycle...")
        try:
            pending = workflow.get_pending_images(drive_client, tracker)
            
            if not pending:
                logger.info("No new images to process")
                return 0
            
            processed = 0
            for file in pending:
                success = workflow.process_single_image(
                    file=file,
                    drive_client=drive_client,
                    image_editor=image_editor,
                    caption_agent=caption_agent,
                    telegram_notifier=telegram_notifier,
                    image_tracker=tracker
                )
                if success:
                    processed += 1
            
            logger.info(f"Processed {processed}/{len(pending)} new image(s)")
            return processed
            
        except Exception as e:
            logger.error(f"Error in poll cycle: {e}")
            telegram_notifier.send_error_notification(f"Error in poll cycle: {str(e)}")
            return 0
    
    interval = int(get_value('POLLING_INTERVAL_MINUTES', app_config.get('polling_interval_minutes', 5), 'POLLING_INTERVAL_MINUTES'))
    
    scheduler = PollingScheduler(interval_minutes=interval, on_poll=poll)
    scheduler.start()


if __name__ == "__main__":
    main()