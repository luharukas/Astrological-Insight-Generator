"""
Module for parsing user input: birth date, time, and location.
"""
from datetime import datetime

def parse_date(date_str: str):
    """
    Parse a date string in YYYY-MM-DD format into a date object.
    Raises ValueError if the format is incorrect.
    """
    return datetime.strptime(date_str, '%Y-%m-%d').date()

def parse_time(time_str: str):
    """
    Parse a time string in HH:MM format into a time object.
    Raises ValueError if the format is incorrect.
    """
    return datetime.strptime(time_str, '%H:%M').time()

def parse_location(location_str: str):
    """
    Validate and normalize the location string.
    Raises ValueError if the location is empty.
    """
    loc = location_str.strip()
    if not loc:
        raise ValueError('Location must not be empty')
    return loc