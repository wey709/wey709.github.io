---
layout: post
title: "ubuntu에서 PostgreSQL ODBC 설치 및 설정 기록 "
date: 2020-08-25
excerpt: "삽질 기록"
tags: [database, configuration]
comments: true
---

순서:
* unixODBC 설치 
* Psql ODBC 드라이버 설치
* ~/.profile, odbc.ini, odbcinst.ini 설정


#### unix ODBC
unix ODBC 드라이버 매니저를 깔기 위해서는 http://www.unixodbc.org/ 에서 설명하는대로 하면 된다. 
```shell
$ wget ftp://ftp.unixodbc.org/pub/unixODBC/unixODBC-2.3.7.tar.gz
```
```shell
$ tar -xzvf unixODBC-2.3.7.tar.gz
```

```shell
$ sudo ./configure  
$ make
$ sudo make install
```
./configure --prefix=/usr/local/unixODBC가 디폴트이고, 설치 디렉토리를 바꾸고 싶으면 prefix 뒤를 조정하면 된다.



#### Psql ODBC
위와 같은 방법으로 설치한다. ref.https://jdbc.postgresql.org/download.html
```shell
$ wget https://jdbc.postgresql.org/download/postgresql-42.2.16.jar
```
```shell
$ tar -xzvf postgresql-42.2.16.jar
```

```shell
$ sudo ./configure  
$ make
$ sudo make install
```


#### ~/.profile 환경변수 설정

```shell
$ vim ~/.profile
```

편집모드에서 아래 두 줄을 추가한다.
```shell
export ODBCSYSINI=/usr/local/etc
export ODBCINI=/usr/local/odbc.ini
```

잘 됐는지 확인한다.
```shell
cat ~/.profile
``` 

#### odbcinst.ini & odbc.ini  설정
ref. http://www.unixodbc.org/odbcinst.html

odbcinst.ini와 odbc.ini는 디폴트로 /usr/local/etc에 있다.

vi로 편집해준다.
```shell
sudo vim /usr/local/etc/odbcinst.ini
```

odbcinst.ini
```shell
[Postgresql ansi]
Description=Postgresql ODBC driver
Driver=/usr/local/lib/psqlodbca.so
Setup=/usr/lib/x86_64-linux-gnu/odbc/libodbcpsqlS.so
FileUsage=1
```

odbc.ini
```shell
[Postgresql-ansi]
Driver=/usr/local/lib/psqlodbca.so
Servername=localhost
Port=5432
Database=db_name
Username=user_name
Password=password
```

```shell
odbcinst -q -d
```

```shell
isql Postgresql-ansi
```


연결 성공
```shell
+---------------------------------------+
| Connected!                            |
|                                       |
| sql-statement                         |
| help [tablename]                      |
| quit                                  |
|                                       |
+---------------------------------------+
```
