
We are using their implementation of xml to baselines as an alternative implementation to makes sure we don't have bugs in our code. This means we have to jump through a few hoops...

### Generate jars (these are already pushed to the rop)
```
apt-get install maven
sh build.sh
```

### Build a folder with ground truth txt files
```
python xml_to_baselines.py ../../../hisdb/TASK-2/validation results/gt_txt results/gt.lst
```

### Sample run with the gt compared to the gt
```
java -jar built_jars/baseline_evaluator.jar results/gt.lst results/gt.lst -no_s
```

There are a lot of useful tools as part of the baseline_evaluator.jar. You can find them in the HowTo.txt. Just replace their jar name with built_jars/baseline_evaluator.jar
