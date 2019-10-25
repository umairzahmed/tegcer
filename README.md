# TEGCER: Targeted Example Generation for Compilation ERrors

## Setup
### Extract dataset
`unzip ./data/input/dataset.zip`

### Ubuntu/Debian packages
`sudo apt install clang`

### Python packages
`pip install --version requirements.txt`

### Clang include path
Set the `pathClangLib` variable in `./src/Base/ConfigFile.py`, to reflect the valid path to your Clang installation's header files directory.


## Running (pre-trained) Tegcer on sample program
`python -m src.run path/to/buggy.c`

### For example:
1. Given the buggy code 

    `cat data/examples/fig1.c`

    ```c
    #include <stdio.h>

    int main() {
        int c, a=3, b=2, i;    
        c = (a-b) (a+b);

        for(i=0, i<c, i++)
            printf("i=" i);
        
        return 0;
    }
    ```

2. With the following compilation errors

    `clang data/examples/fig1.c`

    ```
    data/examples/fig1.c:5:15: error: called object type 'int' is not a function or function pointer
    c = (a-b) (a+b);
        ~~~~~ ^
    data/examples/fig1.c:7:15: warning: expression result unused [-Wunused-value]
        for(i=0, i<c, i++)
                ~^~
    data/examples/fig1.c:7:22: error: expected ';' in 'for' statement specifier
        for(i=0, i<c, i++)
                        ^
    data/examples/fig1.c:7:22: error: expected ';' in 'for' statement specifier
    data/examples/fig1.c:8:21: error: expected ')'
            printf("i=" i);
                        ^
    data/examples/fig1.c:8:15: note: to match this '('
            printf("i=" i);
                ^
    1 warning and 4 errors generated.

    ```

2. Run the below command

    `python -m src.run data/examples/fig1.c`

3. To generate the following example based feedback

    | Line <br> Number | Actual <br> Line | Predicted <br> Class-ID | Predicted <br> Class-Label | Example <br> Before | Example <br> After |
    | --- | --- | --- | --- | --- | --- |
    | 5   | `c = (a-b) (a+b);` | C<sub>29</sub> | E<sub>15</sub> +\* | `float Amount = P (1+((T*R)/100));` | `float Amount = P*(1+(( T*R)/100));` 
    |     |  | C<sub>123</sub> | E<sub>15</sub> +[ +] -( -) | `{ printf("%s", str + count(N-k+1)); }` | `{ printf("%s", str + count[N-k+1]); }`    
    | 7   | `for(i=0, i<c, i++)` | C<sub>10</sub> | E<sub>7</sub> +; -, | `for(i=0, i<n; i++) {` | `for(i=0; i<n; i++) {` |
    |     |                      | C<sub>63</sub> | E<sub>7</sub> +; | `{ for(i=0; i++) {` | `{ for(i=0; ; i++) {` |
    | 8   | `printf("i=" i);` | C<sub>7</sub> | E<sub>1</sub> +, | `printf("%s" str);` | `printf("%s", str); ` |
    |     |                   | C<sub>112</sub> | E<sub>1</sub> +\\" -" -LITERAL_STRING -INVALID | `printf("'a' is not the same as "a"");` | `printf("'a' is not the same as \"a\"");` |

## Training a new model
`python -m src.train`

### Logs
- `./data/output/deepClassify_summary.csv` file contains the summary of all Tegcer training runs.
- `./data/output/currRun_ConfMatrix.csv` file contains the confusion matrix of the last Tegcer training run.
