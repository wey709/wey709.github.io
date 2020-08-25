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


#### unixODBC
```shell
$ sudo apt install unixodbc
```


#### psqlODBC
ref. https://jdbc.postgresql.org/download.html
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
$ cat ~/.profile
``` 

#### odbcinst.ini & odbc.ini  설정
ref. http://www.unixodbc.org/odbcinst.html

odbcinst.ini와 odbc.ini는 디폴트로 /usr/local/etc에 있다.

vi로 편집해준다.
```shell
$ vim /usr/local/etc/odbcinst.ini
```

odbcinst.ini
```console
[Postgresql ansi]
Description=Postgresql ODBC driver
Driver=/usr/local/lib/psqlodbca.so
Setup=/usr/lib/x86_64-linux-gnu/odbc/libodbcpsqlS.so
FileUsage=1
```

odbc.ini
```console
[Postgresql-ansi]
Driver=/usr/local/lib/psqlodbca.so
Servername=localhost
Port=5432
Database=db_name
Username=user_name
Password=password
```

```shell
$ odbcinst -q -d
```

```console
$ isql Postgresql-ansi
```


연결 성공
```console
+---------------------------------------+
| Connected!                            |
|                                       |
| sql-statement                         |
| help [tablename]                      |
| quit                                  |
|                                       |
+---------------------------------------+
```
