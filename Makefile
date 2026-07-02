CFLAGS = -Wall -pg -g3
LDFLAGS = 
CC = g++
LD = g++
EXEC = tausort.x
LIBS =
OBJS = tausort.o 

######### Suffix rules ########################################
.SUFFIXES :    .o .cc

.cc.o:
	$(CC) $(CFLAGS) -c $<

all: 
	$(MAKE) $(OBJS)
	$(LD) -o $(EXEC) $(OBJS) $(LIBS) $(LDFLAGS)

clean:
	rm -f $(EXEC) $(OBJS)

######### Q_rad explorer web app (runs in a local .venv) ######
# Deploy once:   uv sync         (creates ./.venv and installs deps)
# Then:          make start / make stop / make restart / make status
VENV    := .venv
PY      := $(VENV)/bin/python
PIDFILE := .webapp.pid
LOG     := webapp.log
PORT    := 8771

.PHONY: venv start stop restart status

venv:
	uv sync

start:
	@if [ -f $(PIDFILE) ] && kill -0 `cat $(PIDFILE)` 2>/dev/null; then \
		echo "already running (pid `cat $(PIDFILE)`) at http://localhost:$(PORT)"; \
	elif [ ! -x $(PY) ]; then \
		echo "no $(PY) found -- run 'make venv' (or 'uv sync') first"; exit 1; \
	else \
		nohup $(PY) webapp/server.py > $(LOG) 2>&1 & echo $$! > $(PIDFILE); \
		echo "started (pid `cat $(PIDFILE)`) -> http://localhost:$(PORT); logging to $(LOG)"; \
		echo "(first start reads the ODF, ~10-30s, before it serves)"; \
	fi

stop:
	@if [ -f $(PIDFILE) ] && kill -0 `cat $(PIDFILE)` 2>/dev/null; then \
		kill `cat $(PIDFILE)` && echo "stopped (pid `cat $(PIDFILE)`)"; \
	else \
		echo "not running (no live $(PIDFILE))"; \
	fi
	@rm -f $(PIDFILE)

restart:
	@$(MAKE) stop
	@$(MAKE) start

status:
	@if [ -f $(PIDFILE) ] && kill -0 `cat $(PIDFILE)` 2>/dev/null; then \
		echo "running (pid `cat $(PIDFILE)`) at http://localhost:$(PORT)"; \
	else \
		echo "stopped"; \
	fi
