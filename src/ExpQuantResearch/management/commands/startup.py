# startup.py
import asyncio
import MySQLdb
import logging
import threading
from core.config import config
from django.core.management.base import BaseCommand
from core.control import startup

log = logging.getLogger(__name__)

'''
to run test:
$python3 src/manage.py startup
'''
class Command(BaseCommand):
    help = ''

    def handle(self, *args, **options):
        try:
            startup()

        except Exception as err:
            log.warn(f"error starting app: {err}", exc_info=True)


