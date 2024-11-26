import argparse

parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('colors', nargs='*')
args = parser.parse_args()
print(args.colors)
