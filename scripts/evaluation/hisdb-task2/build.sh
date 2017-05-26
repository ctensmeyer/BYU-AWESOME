
git clone https://github.com/Transkribus/TranskribusBaseLineEvaluationScheme.git
cd TranskribusBaseLineEvaluationScheme
wget https://github.com/PRImA-Research-Lab/prima-core-libs/releases/download/v1.2/PrimaBasic.jar
wget https://github.com/PRImA-Research-Lab/prima-core-libs/releases/download/v1.2/PrimaDla.jar
wget https://github.com/PRImA-Research-Lab/prima-core-libs/releases/download/v1.2/PrimaIo.jar
wget https://github.com/PRImA-Research-Lab/prima-core-libs/releases/download/v1.2/PrimaMaths.jar

mvn install:install-file -DgroupId=org.primaresearch -DartifactId=PrimaBasic -Dversion=2017-03-01 -Dpackaging=jar -Dfile=PrimaBasic.jar
mvn install:install-file -DgroupId=org.primaresearch -DartifactId=PrimaMaths -Dversion=2017-03-01 -Dpackaging=jar -Dfile=PrimaMaths.jar
mvn install:install-file -DgroupId=org.primaresearch -DartifactId=PrimaIo -Dversion=2017-03-01 -Dpackaging=jar -Dfile=PrimaIo.jar
mvn install:install-file -DgroupId=org.primaresearch -DartifactId=PrimaDla -Dversion=2017-03-01 -Dpackaging=jar -Dfile=PrimaDla.jar


cp ../xml_cvt_pom.xml pom.xml

#Test cases cause a failure
mvn install -Dmaven.test.skip=true

tar -xvzf TranskribusBaseLineEvaluationScheme_v0.1.1.tar.gz

cd ..
mkdir built_jars

cp TranskribusBaseLineEvaluationScheme/TranskribusBaseLineEvaluationScheme_v0.1.1/TranskribusBaseLineEvaluationScheme-0.1.1-jar-with-dependencies.jar built_jars/baseline_evaluator.jar

cp TranskribusBaseLineEvaluationScheme/target/TranskribusBaseLineEvaluationScheme-0.1.1-jar-with-dependencies.jar built_jars/convert_xml.jar
