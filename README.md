# TEGCER: Targeted Example Generation for Compilation ERrors
TEGCER is an automated feedback generation tool for novice programmers, in the form of examples. TEGCER uses supervised classification to match compilation errors in new code submissions with relevant pre-existing errors, submitted by other students before. 

The dense neural network used to perform this classification task is trained on 15,000+ error-repair code examples. The proposed model yields a test set classification Pred@3 accuracy of 97.7% across 212 error category labels. Using this model as its base, TEGCER presents students with the closest relevant examples of solutions for their specific error on demand. 

A large scale (N > 230) usability study shows that students who use TEGCER are able to resolve errors more than 25% faster on average than students being assisted by human tutors.

## Contributors
- [Umair Z. Ahmed](https://www.cse.iitk.ac.in/users/umair/)<sup>*</sup>, [IIT Kanpur](https://www.cse.iitk.ac.in/)
- [Renuka Sindhgatta](https://staff.qut.edu.au/staff/renuka.sindhgattarajan)<sup>*</sup>, [Queensland University of Technology](https://www.qut.edu.au/)
- [Nisheeth Srivastava](https://www.cse.iitk.ac.in/users/nsrivast/), [IIT Kanpur](https://www.cse.iitk.ac.in/)
- [Amey Karkare](https://www.cse.iitk.ac.in/users/karkare/), [IIT Kanpur](https://www.cse.iitk.ac.in/)

<sup>*</sup> Part of this work was carried out by the author at [IBM Research](https://www.research.ibm.com/labs/india/).

## Publication
If you use any part of our TEGCER tool, then please do cite our [ASE-2019 TEGCER paper](https://arxiv.org/pdf/1909.00769.pdf).

```
@inproceedings{ahmed2019tegcer,
    title={Targeted Example Generation for Compilation Errors},
    author={Ahmed, Umair Z. and Sindhgatta, Renuka and Srivastava, Nisheeth and Karkare, Amey},
    booktitle={The 34th IEEE/ACM International Conference on Automated Software Engineering (ASE 2019)},
    year={2019},
    organization={IEEE/ACM}
}
```

If you use any part of our dataset, then please cite both [ASE-2019 TEGCER paper](https://arxiv.org/pdf/1909.00769.pdf) which released this dataset, as well as [Prutor IDE paper](https://arxiv.org/pdf/1608.03828.pdf) which collated this dataset.

```
@article{das2016prutor,
  title={Prutor: A system for tutoring CS1 and collecting student programs for analysis},
  author={Das, Rajdeep and Ahmed, Umair Z and Karkare, Amey and Gulwani, Sumit},
  journal={arXiv preprint arXiv:1608.03828},
  year={2016}
}
```

## Dataset
Our student code repository consists of code attempts made by students, during the 2015–2016 fall semester course offering of Introductory to C Programming (CS1) at [IIT Kanpur](http://www.iitk.ac.in/), a large public university. This course was credited by 400+ first year undergraduate students, who attempted 40+ different programming assignments as part of course requirement. These assignments were completed on a custom web-browser based IDE [Prutor](https://www.cse.iitk.ac.in/users/karkare/prutor/), which records all intermediate code attempts.

The `./data/input/dataset.zip` archive contains more than 20,000 buggy-correct program pairs, such that (i) the student program failed to compile and (ii) the same student edited a single-line in buggy program to repair it. This dataset is further described in Section-III of our [ASE-2019 TEGCER paper](https://arxiv.org/pdf/1909.00769.pdf).

The column names in extracted `dataset.csv` file are as follows:
- **sourceText**: The buggy program
- **targetText**: The correct/repaired program
- **sourceTime**: Timestamp of buggy program
- **targetTime**: Timestamp of correct program
- **sourceLineText**: The buggy line which was modified/replaced in buggy program
- **targetLineText**: The repaired line
- **sourceLineAbs**: Abstraction of *sourceLineText*
- **targetLineAbs**: Abstraction of *targetLineText*
- **errorClang**: The error message returned by [Clang](https://clang.llvm.org/) C compiler
- **ErrSet**: The error-group (EG) of buggy program
- **errSet_diffs**: The error-repair class that the buggy program belongs to. A combination of *ErrSet* and its repair (diff between *sourceLineAbs* and *targetLineAbs*)
       


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
