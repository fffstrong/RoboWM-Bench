"""
Unified logging configuration utility module.

Provides unified logging configuration supporting both console and file output.
All places that need logging should import the configured logger from this module.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Global log file name for unified logging across all modules in a single run
_global_log_file_name: Optional[str] = None
# Flag to track if global log file name has been auto-initialized
_global_log_file_auto_initialized: bool = False


def get_project_root() -> Path:
    """
    Get the project root directory path.
    
    Searches upward from the current file location until finding a directory
    containing README.md or .git. If not found, goes up 4 levels from the
    current file (from utils/logger.py to project root).
    
    Returns:
        Path: Project root directory path
    """
    current_file = Path(__file__).resolve()
    
    # Search upward from utils/logger.py to find project root
    # utils -> lehome -> lehome -> source -> lehome_challenge
    # Need to go up 4 levels
    potential_root = current_file.parent.parent.parent.parent
    
    # Verify if it's the project root (check for README.md or .git)
    if (potential_root / "README.md").exists() or (potential_root / ".git").exists():
        return potential_root
    
    # If not found, try searching upward from current file
    for parent in current_file.parents:
        if (parent / "README.md").exists() or (parent / ".git").exists():
            return parent
    
    # If still not found, return default path (4 levels up)
    return potential_root


def get_running_script_name() -> str:
    """
    Get the name of the currently running Python script.
    
    Returns:
        str: Script name without extension, or 'run' if cannot determine
    """
    if len(sys.argv) > 0 and sys.argv[0]:
        script_path = Path(sys.argv[0])
        # Get script name without extension
        script_name = script_path.stem
        # If it's '__main__', try to get from __main__ module
        if script_name == '__main__' or script_name == '':
            try:
                import __main__
                if hasattr(__main__, '__file__'):
                    script_path = Path(__main__.__file__)
                    script_name = script_path.stem
            except:
                pass
        return script_name if script_name else 'run'
    return 'run'


def generate_log_filename(script_name: Optional[str] = None) -> str:
    """
    Generate log filename with timestamp and script name.
    
    Format: MM-DD_HH-MM-SS_scriptname.log
    Example: 10-27_14-30-22_logger.log
    
    Args:
        script_name: Script name. If None, gets from get_running_script_name()
    
    Returns:
        str: Generated log filename
    """
    if script_name is None:
        script_name = get_running_script_name()
    
    # Get current timestamp
    now = datetime.now()
    timestamp = now.strftime("%m-%d_%H-%M-%S")
    
    # Generate filename: MM-DD_HH-MM-SS_scriptname.log
    log_filename = f"{timestamp}_{script_name}.log"
    
    return log_filename


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_file_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    format_string: Optional[str] = None,
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name, typically use __name__. If None, uses the calling module's name
        level: Logging level, defaults to logging.INFO
        log_to_file: Whether to write logs to file, defaults to True
        log_file_name: Log file name. If None, generates filename with timestamp and script name
            (format: MM-DD_HH-MM-SS_scriptname.log, e.g., 10-27_14-30-22_logger.log)
        log_dir: Log file directory. If None, uses logs folder under project root
        format_string: Log format string. If None, uses default format
        date_format: Date format string, defaults to "%Y-%m-%d %H:%M:%S"
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get logger
    if name is None:
        # Get the calling module's name
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            name = caller_frame.f_globals.get('__name__', 'lehome')
        finally:
            del frame
    
    # If name is __main__, use script name instead for better readability
    if name == '__main__':
        script_name = get_running_script_name()
        # Use script name as logger name (e.g., 'run' instead of '__main__')
        name = script_name if script_name != 'run' else 'main'
    
    logger = logging.getLogger(name)
    
    # Use global log file name if set and not explicitly overridden
    if log_file_name is None and _global_log_file_name is not None:
        log_file_name = _global_log_file_name
    
    # If logger already has handlers, update level if needed (avoid duplicate configuration)
    # Note: To ensure all modules use the same log file, set global log file name
    # before importing modules that create loggers.
    if logger.handlers:
        # Update level for existing handlers if a different level is requested
        if logger.level != level:
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
        return logger
    
    # Set logging level
    logger.setLevel(level)
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # Console output handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File output handler (optional)
    if log_to_file:
        # Determine log directory
        if log_dir is None:
            project_root = get_project_root()
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir)
        
        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine log file name
        # Priority: 1. Explicit log_file_name (from kwargs), 2. Global log file name, 3. Generate new
        if log_file_name is None:
            # Generate filename with timestamp and script name
            log_file_name = generate_log_filename()
        
        log_file = log_dir / log_file_name
        
        # File output handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False
    
    return logger


def set_global_log_file_name(log_file_name: str) -> None:
    """
    Set a global log file name that will be used by all loggers in the current run.
    
    This allows all modules to write to the same log file, making it easier to
    track all output from a single execution.
    
    IMPORTANT: Call this function BEFORE importing any modules that use get_logger()
    if you want to use a custom log file name. Otherwise, the first call to get_logger()
    will auto-generate a log filename based on the running script name.
    
    Args:
        log_file_name: Log file name to use for all loggers (e.g., "01-02_22-10-47_load_env.log")
    
    Example:
        >>> # Set custom log filename before importing other modules
        >>> from lehome.utils.logger import set_global_log_file_name
        >>> set_global_log_file_name("custom_name.log")
        >>> from lehome.utils.logger import get_logger
        >>> logger = get_logger(__name__)  # Will use the custom log file name
    """
    global _global_log_file_name, _global_log_file_auto_initialized
    _global_log_file_name = log_file_name
    _global_log_file_auto_initialized = True  # Mark as initialized to prevent auto-generation


def get_global_log_file_name() -> Optional[str]:
    """
    Get the current global log file name.
    
    Returns:
        Optional[str]: Current global log file name, or None if not set
    """
    return _global_log_file_name


def _auto_initialize_global_log_file_name() -> None:
    """
    Automatically initialize global log file name based on running script name.
    
    This function is called automatically on first logger creation if no global
    log file name has been set. It generates a log filename with timestamp and
    script name, ensuring all modules in the same run use the same log file.
    
    Format: MM-DD_HH-MM-SS_scriptname.log
    Example: 10-27_14-30-22_load_env.log
    """
    global _global_log_file_name, _global_log_file_auto_initialized
    
    # Only auto-initialize once per run
    if _global_log_file_auto_initialized:
        return
    
    # If global log file name is already set manually, don't override
    if _global_log_file_name is not None:
        _global_log_file_auto_initialized = True
        return
    
    # Auto-generate log filename based on running script
    script_name = get_running_script_name()
    log_file_name = generate_log_filename(script_name)
    _global_log_file_name = log_file_name
    _global_log_file_auto_initialized = True


def get_logger(name: Optional[str] = None, **kwargs) -> logging.Logger:
    """
    Get a configured logger instance (convenience function).
    
    This is a convenience wrapper around setup_logger using default configuration.
    If the logger has already been configured, returns the existing instance directly.
    
    On first call, if no global log file name has been set, it will automatically
    generate one based on the running script name (format: MM-DD_HH-MM-SS_scriptname.log).
    This ensures all modules in the same run write to the same log file.
    
    If a global log file name has been set via set_global_log_file_name(), it will
    be used unless log_file_name is explicitly provided in kwargs.
    
    Args:
        name: Logger name, typically use __name__
        **kwargs: Additional arguments passed to setup_logger
    
    Returns:
        logging.Logger: Configured logger instance
    
    Example:
        >>> from lehome.utils.logger import get_logger
        >>> logger = get_logger(__name__)  # Auto-generates log filename on first call
        >>> logger.info("This is a log message")
        
        >>> # Or manually set log filename before importing other modules:
        >>> from lehome.utils.logger import set_global_log_file_name, get_logger
        >>> set_global_log_file_name("custom_name.log")
        >>> logger = get_logger(__name__)  # Uses custom name
    """
    # Auto-initialize global log file name on first call if not set
    _auto_initialize_global_log_file_name()
    
    # Use global log file name if set and not explicitly overridden
    if 'log_file_name' not in kwargs and _global_log_file_name is not None:
        kwargs['log_file_name'] = _global_log_file_name
    
    return setup_logger(name=name, **kwargs)


if __name__ == "__main__":
    # Test the logger
    logger = get_logger(__name__)
    logger.info("This is a log message")
