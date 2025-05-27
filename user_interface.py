#!/usr/bin/env python3
"""
User interface utilities for the stock dashboard runner
"""
from typing import Optional


class UserInterface:
    """Handles user interaction and console output"""
    
    @staticmethod
    def print_header(title: str, width: int = 30) -> None:
        """
        Print a formatted header
        
        Args:
            title: Title to display
            width: Width of the separator line
        """
        print(title)
        print("=" * width)
    
    @staticmethod
    def print_success(message: str) -> None:
        """Print a success message with checkmark"""
        print(f"✅ {message}")
    
    @staticmethod
    def print_error(message: str) -> None:
        """Print an error message with X mark"""
        print(f"❌ {message}")
    
    @staticmethod
    def print_info(message: str) -> None:
        """Print an informational message"""
        print(f"ℹ️  {message}")
    
    @staticmethod
    def print_warning(message: str) -> None:
        """Print a warning message"""
        print(f"⚠️  {message}")
    
    @staticmethod
    def get_yes_no_input(prompt: str, default: bool = True) -> bool:
        """
        Get yes/no input from user
        
        Args:
            prompt: Question to ask the user
            default: Default value if user just presses Enter
            
        Returns:
            bool: True for yes, False for no
        """
        default_text = "Y/n" if default else "y/N"
        user_input = input(f"{prompt} ({default_text}): ").lower().strip()
        
        if not user_input:
            return default
        
        return user_input.startswith('y')
    
    @staticmethod
    def get_user_choice(prompt: str, 
                       choices: dict, 
                       default: Optional[str] = None) -> str:
        """
        Get user choice from a set of options
        
        Args:
            prompt: Question to ask the user
            choices: Dictionary of choice_key: description
            default: Default choice if user just presses Enter
            
        Returns:
            str: The chosen key
        """
        print(prompt)
        for key, description in choices.items():
            marker = " (default)" if key == default else ""
            print(f"  {key}: {description}{marker}")
        
        while True:
            user_input = input("Choice: ").strip()
            
            if not user_input and default:
                return default
            
            if user_input in choices:
                return user_input
            
            print(f"Invalid choice. Please choose from: {', '.join(choices.keys())}")
    
    @staticmethod
    def print_separator(char: str = "-", width: int = 50) -> None:
        """Print a separator line"""
        print(char * width)