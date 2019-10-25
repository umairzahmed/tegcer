# TEGCER: Targeted Example Generation for Compilation ERrors
TEGCER is an automated example based feedback generation tool for novice programmers. TEGCER uses supervised classification to match compilation errors in new code submissions with relevant pre-existing errors, submitted by other students before. 

The dense neural network used to perform this classification task is trained on 15000+ error-repair code examples. The proposed model yields a test set classification Pred@3 accuracy of 97.7% across 212 error category labels. Using this model as its base, TEGCER presents students with the closest relevant examples of solutions for their specific error on demand. 

A large scale (N > 230) usability study shows that students who use TEGCER are able to resolve errors more than 25% faster on average than students being assisted by human tutors.

## Contributors
- [Umair Z. Ahmed](https://www.cse.iitk.ac.in/users/umair/)<sup>*</sup>, [IIT Kanpur](https://www.cse.iitk.ac.in/)
- [Renuka Sindhgatta](https://staff.qut.edu.au/staff/renuka.sindhgattarajan)<sup>*</sup>, [Queensland University of Technology](https://www.qut.edu.au/)
- [Nisheeth Srivastava](https://www.cse.iitk.ac.in/users/nsrivast/), [IIT Kanpur](https://www.cse.iitk.ac.in/)
- [Amey Karkare](https://www.cse.iitk.ac.in/users/karkare/), [IIT Kanpur](https://www.cse.iitk.ac.in/)

\* Part of this work was carried out by the author at [IBM Research](https://www.research.ibm.com/labs/india/).

## Publication
If you use any part of our TEGCER tool or data present in this repository, then please do cite our [ASE-2019 TEGCER paper](https://arxiv.org/pdf/1909.00769.pdf).

```
@inproceedings{ahmed2019tegcer,
    title={Targeted Example Generation for Compilation Errors},
    author={Ahmed, Umair Z. and Sindhgatta, Renuka and Srivastava, Nisheeth and Karkare, Amey},
    booktitle={The 34th IEEE/ACM International Conference on Automated Software Engineering (ASE 2019)},
    year={2019},
    organization={IEEE/ACM}
}
```

## Setup
### Extract dataset
`unzip ./data/input/dataset.zip`

### Ubuntu/Debian packages
`sudo apt install clang`

### Python packages
`pip install --version requirements.txt`

### Clang include path
Set the `pathClangLib` variable in `./src/Base/ConfigFile.py`, to reflect the valid path to your Clang installation's header files directory.


## Running (pre-trained) Tegcer on buggy program
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

| Pred@1 | Pred@3 | Pred@5 |
| --- | --- | --- |
| 87.06% | 97.68% | 98.81% |

### Logs
- `./data/output/deepClassify_summary.csv` file contains the summary of all Tegcer training runs.
- `./data/output/currRun_ConfMatrix.csv` file contains the confusion matrix of the last Tegcer training run.
