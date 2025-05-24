import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def cleanup_old_files(directory, max_age_days=7):
    """Remove files older than max_age_days from the specified directory."""
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return 0
    
    removed_count = 0
    now = datetime.now()
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = now - datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_age.days > max_age_days:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        logger.info(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {e}")
        
        logger.info(f"Cleanup completed: removed {removed_count} files older than {max_age_days} days from {directory}")
        return removed_count
    except Exception as e:
        logger.error(f"Error during cleanup of directory {directory}: {e}")
        return 0

def limit_file_count(directory, max_files=1000):
    """Ensure no more than max_files are kept in the directory (removes oldest first)."""
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return 0
    
    removed_count = 0
    
    try:
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                files.append((file_path, os.path.getmtime(file_path)))
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Remove oldest files if we exceed the limit
        files_to_remove = files[:-max_files] if len(files) > max_files else []
        
        for file_path, _ in files_to_remove:
            try:
                os.remove(file_path)
                removed_count += 1
                logger.info(f"Removed excess file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")
        
        logger.info(f"File count limit enforced: removed {removed_count} oldest files from {directory}, keeping max {max_files}")
        return removed_count
    except Exception as e:
        logger.error(f"Error during file count limiting in directory {directory}: {e}")
        return 0

def perform_cleanup(directory, max_age_days=7, max_files=1000):
    """Perform both time-based and count-based cleanup."""
    time_removed = cleanup_old_files(directory, max_age_days)
    count_removed = limit_file_count(directory, max_files)
    return time_removed + count_removed
