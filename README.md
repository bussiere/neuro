<p align="center">
<a href="https://tensortask.com">
<img width="200" alt="TensorTask Logo" src="https://storage.googleapis.com/tensortask-static/tensortask_transparent.png">
</a>
</p>

[![GoDoc][1]][2] [![Go Report Card][3]][4] [![Keybase Chat][5]][6] [![Cloud Build][7]][8]

[1]: https://godoc.org/github.com/tensortask/neuro?status.svg
[2]: https://godoc.org/github.com/tensortask/neuro
[3]: https://goreportcard.com/badge/github.com/tensortask/neuro
[4]: https://goreportcard.com/report/github.com/tensortask/neuro
[5]: https://img.shields.io/badge/keybase%20chat-tensortask.public-blue.svg
[6]: https://keybase.io/team/tensortask.public
[7]: https://storage.googleapis.com/tensortask-static/build/neuro.svg
[8]: https://github.com/sbsends/cloud-build-badge

[9]: https://github.com/golang/go/wiki/Modules
[10]: https://github.com/golang/go/wiki/Modules#how-to-install-and-activate-module-support
[11]: https://www.tensorflow.org/install/lang_go
[12]: https://www.tensorflow.org/install/lang_c

# 🧠 Neuro: TensorFlow + Go

```diff
- #############################
- #       ALPHA PACKAGE       #
- #############################
```

## Supported Platforms

* Linux, 64-bit, x86
* macOS X, Version 10.12.6 (Sierra) or higher

## Dependencies

#### Golang Dependencies

Neuro depends on a number of [external Golang dependencies](./go.sum), and uses [go modules][9] to manage them. Go modules are currently an experimental opt-in feature in Go 1.11 (see installation instructions [here][10].

To satisfy dependencies: 
1) ensure that `GO111MODULE=on` and `go version` >= 1.11
2) make sure that your project root / GOPATH are correct.

### TensorFlow Dependencies

Neuro depends on the TensorFlow Golang package. Follow instructions to install the Golang package [here][11]. 

The Golang package depends on the TensorFlow C library. Install instructions can be found [here][12].

#### Mac Users

System Integrity Protection (SIP) is a security feature of macOS originally introduced in OS X El Capitan. SIP disables write access to system level directories. In order install the TensorFlow C library, you must either install the library to a different location and update the linker or disable SIP.

##### Custom Directory + Updated Linker
`export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/customTensorFlowInstallDir/lib`

##### Disabling SIP

1) Boot into recovery mode (hold Command-R on start up)
2) Enter terminal (in recovery mode)
3) Enter `csrutil disable; reboot`
4) Unzip tf C library tar into `/usr/local`

e.g. `curl -L https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-..<VERSION>.tar.gz | sudo tar -C /usr/local -xz`
