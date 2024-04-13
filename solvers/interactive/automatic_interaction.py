import argparse
import pexpect
import traceback
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, type=str)
    parser.add_argument("-f", default=None, type=str)
    args = parser.parse_args()

    assert (args.file is not None) or (args.f is not None)

    filename = args.file if(args.file is not None) else args.f

    with open(filename, 'r') as f:
        program = pexpect.spawn('python main.py', encoding='utf-8')
        program.logfile = sys.stdout#.buffer
        
        line = f.readline()
        try:
            while (line != ''):
                program.expect('Charging requests:')
                program.sendline(line.replace('\n', ''))
                line = f.readline()
                program.expect('Power allocations:')
                program.sendline(line.replace('\n', ''))
                line = f.readline()
        except Exception as e:
            print(traceback.format_exc())
            pass
        program.sendline(chr(3))
        program.expect(pexpect.EOF)
