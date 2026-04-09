import logging

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Error durante el procesamiento de una imagen."""
    pass


def process_single_image(
    file: dict,
    drive_client,
    image_editor,
    caption_agent,
    telegram_notifier,
    image_tracker
) -> bool:
    """
    Procesa una sola imagen.
    
    Returns:
        True si se procesó correctamente, False si hubo error.
    """
    file_id = file['id']
    file_name = file['name']
    
    logger.info(f"Processing image: {file_name}")
    
    try:
        image_content = drive_client.download_image(file_id)
        
        edited_image = image_editor.edit_image(image_content)
        
        caption = caption_agent.generate_caption(edited_image)
        
        output_file_name = f"processed_{file_name}"
        drive_client.upload_image(
            drive_client.output_folder_id,
            output_file_name,
            edited_image
        )
        
        telegram_notifier.send_image_with_caption(edited_image, caption)
        
        image_tracker.mark_processed(file_id)
        
        logger.info(f"Successfully processed image: {file_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing image {file_name}: {e}")
        telegram_notifier.send_error_notification(
            f"Error processing {file_name}: {str(e)}"
        )
        return False


def get_pending_images(drive_client, image_tracker) -> list[dict]:
    """
    Obtiene las imágenes nuevas que aún no han sido procesadas.
    """
    all_images = drive_client.list_images(drive_client.input_folder_id)
    
    pending = []
    for image in all_images:
        if not image_tracker.is_processed(image['id']):
            pending.append(image)
        else:
            logger.debug(f"Skipping already processed: {image['name']}")
    
    return pending


def get_processed_count(pending_images: list[dict]) -> int:
    """Retorna el número de imágenes procesadas."""
    return len(pending_images)