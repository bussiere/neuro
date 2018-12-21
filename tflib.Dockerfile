# GOLANG BASE IMAGE -- MUST VERSION
FROM golang:1.11.1-stretch

# DOWNLOAD TF C LIBRARY
RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz
# UNZIP TF C LIBRARY TO /USR/LOCAL
RUN tar -xz -f libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz -C /usr/local && rm libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz
# UPDATE LINKER
RUN ldconfig

# THIS IMAGE IS USED IN THE CLOUD BUILD TESTING STEP. INCLUDES TF C LIBRARY.