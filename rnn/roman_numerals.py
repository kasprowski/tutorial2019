'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

Class that converts arabic and roman numbers to each other

taken from: 
https://gist.github.com/riverrun/ac91218bb1678b857c12
'''

class ToRoman(int):
    def __new__(cls, number):
        if number > 3999:
            raise ValueError('Values over 3999 are not allowed: {}'.format(number))
        if number < 0:
            raise ValueError('Negative values are not allowed: {}'.format(number))
        return super().__new__(cls, number)

    def __init__(self, number):
        to_roman = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
                6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X', 20: 'XX',
                30: 'XXX', 40: 'XL', 50: 'L', 60: 'LX', 70: 'LXX', 80: 'LXXX',
                90: 'XC', 100: 'C', 200: 'CC', 300: 'CCC', 400: 'CD', 500: 'D',
                600: 'DC', 700: 'DCC', 800: 'DCCC', 900: 'CM', 1000: 'M',
                2000: 'MM', 3000: 'MMM'}
        self.roman = ''.join([to_roman.get(num) for num in self][::-1])

    def __iter__(self):
        number = self.__str__()
        count = 1
        for digit in number[::-1]:
            if digit != '0':
                yield int(digit) * count
            count *= 10

class ToArabic(str):
    def __init__(self, roman):
        roman = self.check_valid(roman)
        keys = ['IV', 'IX', 'XL', 'XC', 'CD', 'CM', 'I', 'V', 'X', 'L', 'C', 'D', 'M']
        to_arabic = {'IV': '4', 'IX': '9', 'XL': '40', 'XC': '90', 'CD': '400', 'CM': '900',
                'I': '1', 'V': '5', 'X': '10', 'L': '50', 'C': '100', 'D': '500', 'M': '1000'}
        for key in keys:
            if key in roman:
                roman = roman.replace(key, ' {}'.format(to_arabic.get(key)))
        self.arabic = sum(int(num) for num in roman.split())

    def check_valid(self, roman):
        roman = roman.upper()
        invalid = ['IIII', 'VV', 'XXXX', 'LL', 'CCCC', 'DD', 'MMMM']
        if any(sub in roman for sub in invalid):
            raise ValueError('Numerus invalidus est: {}'.format(roman))
        return roman

def convert(number):
    if isinstance(number, int):
        num = ToRoman(number)
        return num.roman
    num = ToArabic(number)
    return num.arabic
