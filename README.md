# TF3P

Three-dimensional force fields fingerprints

The trained models are available on https://disk.pku.edu.cn/link/D80390F955C4DC9104377ED09B76DB57

Ref: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00005


Dependecies: 
```
pip install tables fire
conda install pytorch-scatter -c pyg
```

## Mod by Ruibo for the QSAR project:

`inference.py` is an interface to generate TF3P embeddings. 

Usage (main_for_topo_project):

```bash
for eachfile in $yourfilenames
do
    echo "TF3P: $eachfile"
    python inference.py -wd $wd/$eachfile
done
```

A numpy array will be picked to the WD or the designated file path. 

`num_digit_caps` is set to 1024. Modify the file if needed. 

`main_for_topo_project()` is optimized for the QSAR project. Given a WD it will search for the SMILE txt. If a more general function is needed, uncomment the `main()` in the inference.py. 

