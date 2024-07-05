import os
import re
from typing import TypeVar, Callable, List

T = TypeVar('T')  # Typ generyczny dla wejścia

class InputConverter:
    def __init__(self, input_data: T):
        self.input = input_data

    def convert_by(self, *converters: Callable[[T], T]) -> T:
        result = self.input
        for converter in converters:
            result = converter(result)
        return result


def read_lines(path: str) -> List[str]:
    with open(path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def join_lines(lines: List[str]) -> str:
    return ''.join(lines)

def extract_integers(s: str) -> List[int]:
    return [int(num) for num in re.findall(r'-?\d+', s)]

def calculate_sum(numbers: List[int]) -> int:
    return sum(numbers)

def main():
    # test with: python input_converter.py "Warszawa 100 Kielce 200 Szczecin 300"

    file_conv = InputConverter("lam-com-file.txt")

    lines = file_conv.convert_by(read_lines)
    text = file_conv.convert_by(read_lines, join_lines)
    ints = file_conv.convert_by(read_lines, join_lines, extract_integers)
    sum_ints = file_conv.convert_by(read_lines, join_lines, extract_integers, calculate_sum)

    print(lines)
    print(text)
    print(ints)
    print(sum_ints)

    import sys
    arglist = sys.argv[1:]
    slist_conv = InputConverter(arglist)
    sum_ints = slist_conv.convert_by(join_lines, extract_integers, calculate_sum)
    print(sum_ints)

if __name__ == "__main__":
    main()
