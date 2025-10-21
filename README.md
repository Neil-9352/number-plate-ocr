# number-plate-ocr

## generate certificate files
```bash
$ mkdir certs
$ cd certs
$ openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```
# update readme for test commit