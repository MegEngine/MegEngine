# The main purpose of this folder is to adjust the compilation order of megdnn to improve the overall compilation time.If you add a new opr and find that it significantly slows down the compilation time, you can manually add the opr to the front of priority_compile_opr.txt. If you add more opr, you can also run: python3 sort_compile_time_map.py. However, this operation will be very time-consuming because it is a single-threaded compilation.

- priority_compile_opr.txt: Compile order sorted by compile time
- sort_compile_time_map.py: Generate the compile script for the above two files
## Usage
```bash
python3 sort_compile_time_map.py
```

