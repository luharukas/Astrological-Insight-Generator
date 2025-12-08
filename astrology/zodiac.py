"""
Module for determining zodiac sign from birth date.
"""
from datetime import date

def get_zodiac_sign(birth_date: date) -> str:
    """
    Determine the zodiac sign for a given birth date (sun sign).
    Returns the sign as a string.
    """
    month = birth_date.month
    day = birth_date.day
    # Aries: March 21 – April 19
    if (month == 3 and day >= 21) or (month == 4 and day <= 19):
        return 'Aries'
    # Taurus: April 20 – May 20
    if (month == 4 and day >= 20) or (month == 5 and day <= 20):
        return 'Taurus'
    # Gemini: May 21 – June 20
    if (month == 5 and day >= 21) or (month == 6 and day <= 20):
        return 'Gemini'
    # Cancer: June 21 – July 22
    if (month == 6 and day >= 21) or (month == 7 and day <= 22):
        return 'Cancer'
    # Leo: July 23 – August 22
    if (month == 7 and day >= 23) or (month == 8 and day <= 22):
        return 'Leo'
    # Virgo: August 23 – September 22
    if (month == 8 and day >= 23) or (month == 9 and day <= 22):
        return 'Virgo'
    # Libra: September 23 – October 22
    if (month == 9 and day >= 23) or (month == 10 and day <= 22):
        return 'Libra'
    # Scorpio: October 23 – November 21
    if (month == 10 and day >= 23) or (month == 11 and day <= 21):
        return 'Scorpio'
    # Sagittarius: November 22 – December 21
    if (month == 11 and day >= 22) or (month == 12 and day <= 21):
        return 'Sagittarius'
    # Capricorn: December 22 – January 19
    if (month == 12 and day >= 22) or (month == 1 and day <= 19):
        return 'Capricorn'
    # Aquarius: January 20 – February 18
    if (month == 1 and day >= 20) or (month == 2 and day <= 18):
        return 'Aquarius'
    # Pisces: February 19 – March 20
    if (month == 2 and day >= 19) or (month == 3 and day <= 20):
        return 'Pisces'
    raise ValueError(f'Invalid birth date: {birth_date}')