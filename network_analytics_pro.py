#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         MULTILINGUAL ACTOR-EVENT NETWORK DATABASE SYSTEM                     ║
║                                                                              ║
║  For analyzing hidden economical, political, or criminal networks            ║
║  Support Languages: Chinese (PRC/HK/SG/TW), Russian, English names           ║
║  Features: Temporal tracking, event analysis                                 ║
║                                                                              ║
║  Version: 1.0 (Production Ready)                                             ║
║  Platform: Python 3.8+, SQLite3, Tkinter                                     ║
║  Author: Zonengeist                                                          ║
║  Date: 2025-11-08                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALLATION:
  python3 network_analytics_pro.py

DEPENDENCIES:
  • Python 3.8+
  • sqlite3 (built-in)
  • tkinter (built-in)
  • difflib (built-in)
  • uuid (built-in)
  • json (built-in)
  • datetime (built-in)

DATABASE SQLite:
  • File: network_analytics.db
  • Tables: 5 core + support tables
  • Auto-initialized on first run

"""

import sqlite3
import json
import uuid
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from datetime import datetime, date
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field, asdict
import difflib
import logging
import os

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

DB_PATH = "network_analytics.db"
LOG_FILE = "network_analytics.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS (Session Context)
# ============================================================================

@dataclass
class SessionContext:
    """Manages application session state"""
    active_actor_id: Optional[str] = None
    active_event_id: Optional[int] = None
    unsaved_changes: Dict[str, bool] = field(default_factory=lambda: {
        'Actors': False,
        'Edges': False,
        'Events': False,
        'EventParticipation': False
    })
    current_tab: str = 'Actors'
    edit_mode: str = 'new'  # 'new' | 'edit' | 'view'
    
    def clear(self):
        """Reset session state"""
        self.active_actor_id = None
        self.active_event_id = None
        self.unsaved_changes = {k: False for k in self.unsaved_changes}
        self.current_tab = 'Actors'
        self.edit_mode = 'new'
        logger.info("Session cleared")
    
    def mark_unsaved(self, table: str):
        """Mark a table as having unsaved changes"""
        if table in self.unsaved_changes:
            self.unsaved_changes[table] = True
            logger.debug(f"Marked {table} as unsaved")
    
    def clear_unsaved(self, table: str = None):
        """Clear unsaved changes flag"""
        if table:
            if table in self.unsaved_changes:
                self.unsaved_changes[table] = False
        else:
            self.unsaved_changes = {k: False for k in self.unsaved_changes}
        logger.debug(f"Cleared unsaved for {table or 'all'}")
    
    def has_unsaved(self) -> bool:
        """Check if any changes are unsaved"""
        return any(self.unsaved_changes.values())


@dataclass
class LookupResult:
    """Result from name lookup operation"""
    status: str  # 'found' | 'ambiguous' | 'not_found' | 'error'
    actor_id: Optional[str] = None
    candidates: List[Dict] = field(default_factory=list)
    message: str = ""
    confidence: float = 0.0


# ============================================================================
# LAYER 1: DATABASE MANAGEMENT
# ============================================================================

class NetworkDatabase:
    """Database management and CRUD operations"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.initialize_db()
        logger.info(f"Database initialized: {db_path}")
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self.conn
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute SELECT query"""
        if not self.conn:
            self.connect()
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query"""
        if not self.conn:
            self.connect()
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            logger.debug(f"Executed: {query[:50]}...")
            return cursor.lastrowid
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Insert error: {e}")
            raise
    
    def initialize_db(self):
        """Create all tables if not exist"""
        self.connect()
        cursor = self.conn.cursor()
        
        # TABLE 1: actors_id (IMMUTABLE REGISTRY)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS actors_id (
                actor_id TEXT PRIMARY KEY,
                nationality TEXT NOT NULL CHECK(nationality IN ('ru', 'zh_prc', 'zh_hk', 'zh_sg', 'zh_tw', 'other')),
                
                name_ru TEXT UNIQUE,
                name_zh_prc TEXT UNIQUE,
                name_zh_hk TEXT UNIQUE,
                name_zh_sg TEXT UNIQUE,
                name_zh_tw TEXT UNIQUE,
                name_en TEXT UNIQUE,
                
                pinyin TEXT,
                born_year_month TEXT CHECK(born_year_month IS NULL OR born_year_month GLOB '[0-9][0-9]-[0-9][0-9][0-9][0-9]'),
                
                verification_status TEXT DEFAULT 'unverified' CHECK(verification_status IN ('unverified', 'verified', 'rejected')),
                created_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                
                UNIQUE(actor_id),
                CHECK(
                    name_ru IS NOT NULL OR name_zh_prc IS NOT NULL OR
                    name_zh_hk IS NOT NULL OR name_zh_sg IS NOT NULL OR
                    name_zh_tw IS NOT NULL OR name_en IS NOT NULL
                )
            )
        ''')
        
        # Indexes for actors_id
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actors_id_name_ru ON actors_id(name_ru)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actors_id_name_zh_prc ON actors_id(name_zh_prc)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actors_id_name_zh_hk ON actors_id(name_zh_hk)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actors_id_pinyin ON actors_id(pinyin)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actors_id_nationality ON actors_id(nationality)')
        
        # TABLE 2: Actors (BIOGRAPHICAL TIME-SERIES)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Actors (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id TEXT NOT NULL REFERENCES actors_id(actor_id),
                
                observation_date DATE NOT NULL,
                record_status TEXT DEFAULT 'current' CHECK(record_status IN ('current', 'superseded', 'rejected')),
                
                nationality TEXT,
                birth_place TEXT,
                education TEXT,
                academic_titles TEXT,
                academic_works TEXT,
                political_party TEXT,
                party_rank TEXT,
                political_works TEXT,
                prof_position TEXT,
                prof_rank TEXT,
                participation_in_elections TEXT,
                status_in_elected_bodies TEXT,
                affiliation TEXT,
                
                first_seen DATE,
                last_seen DATE,
                
                status TEXT CHECK(status IN ('active', 'inactive', 'retired', 'expelled', 'missing', 'died')),
                
                source TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(actor_id, observation_date)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actors_actor_date ON Actors(actor_id, observation_date DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actors_status ON Actors(status)')
        
        # TABLE 3: Actor_Aliases
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Actor_Aliases (
                alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id TEXT NOT NULL REFERENCES actors_id(actor_id),
                
                alias_name TEXT NOT NULL,
                alias_type TEXT NOT NULL CHECK(alias_type IN ('transliteration', 'historical', 'nickname', 'variant')),
                language_code TEXT,
                confidence REAL DEFAULT 0.8 CHECK(confidence BETWEEN 0.0 AND 1.0),
                source TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_aliases_name ON Actor_Aliases(alias_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_aliases_actor ON Actor_Aliases(actor_id)')
        
        # TABLE 4: Events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_title_ru TEXT NOT NULL,
                event_title_zh TEXT,
                event_title_en TEXT,
                event_date DATE NOT NULL,
                event_date_precision TEXT DEFAULT 'day' CHECK(event_date_precision IN ('day', 'month', 'year', 'unknown')),
                
                description_ru TEXT,
                description_zh TEXT,
                description_en TEXT,
                
                -- Level 1: Domain of Origin
                event_domain TEXT DEFAULT 'unknown' CHECK (event_domain IN (
                    'politics', 'economics', 'society', 'security',
                    'technology', 'healthcare', 'culture', 'sports',
                    'nature', 'unknown'
                )),
                
                -- Level 2: Impact and Nature Type
                event_type TEXT DEFAULT 'unknown' CHECK (event_type IN (
                    'trigger', 'process', 'institutional', 'normative',
                    'symbol_narrative', 'forecast', 'unknown'
                )),
                
                -- Level 3: Geographic Scale
                event_scale TEXT DEFAULT 'unknown' CHECK (event_scale IN (
                    'local', 'national', 'regional', 'global', 'unknown'
                )),
                
                -- Level 4: Impact and Importance
                event_priority TEXT DEFAULT 'unknown' CHECK (event_priority IN (
                    'strategic', 'tactical', 'noise', 'unknown'
                )),
                
                participants_count INTEGER,
                
                sources TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_domain ON events(event_domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_scale ON events(event_scale)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_priority ON events(event_priority)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_precision ON events(event_date_precision)')

        
        # TABLE 5: Event_Participation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_participation (
                participation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL REFERENCES events(event_id),
                actor_id TEXT NOT NULL REFERENCES actors_id(actor_id),
                
                participation_role TEXT DEFAULT 'unknown' CHECK(participation_role IN (
                    'organizer', 'executor', 'mediator', 'beneficiary', 
                    'affected_party', 'provocateur', 'observer', 'unknown'
                )),
                
                participation_function TEXT DEFAULT 'unknown' CHECK(participation_function IN (
                    'leader', 'spokesperson', 'expert', 'witness', 
                    'information_source', 'unknown'
                )),
                
                participation_position TEXT DEFAULT 'unknown' CHECK(participation_position IN (
                    'state_actor', 'commercial_actor', 'non_commercial_actor', 
                    'international_actor', 'illegal_actor', 'religious_actor', 
                    'individual_actor', 'unknown'
                )),
                
                information_role TEXT DEFAULT 'unknown' CHECK(information_role IN (
                    'information_spreader', 'information_target', 
                    'narrative_creator', 'unknown'
                )),
                
                description TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(event_id, actor_id)
            )
        ''')

        # Indexes for performance  
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ep_event ON event_participation(event_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ep_actor ON event_participation(actor_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ep_role ON event_participation(participation_role)')

        # TABLE 6: Edges/Relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_actor_id TEXT NOT NULL REFERENCES actors_id(actor_id),
                target_actor_id TEXT NOT NULL REFERENCES actors_id(actor_id),
                relationship_type TEXT NOT NULL CHECK(
                    relationship_type IN (
                        'kinship', 'trust', 'community', 'informal',
                        'official', 'mentoring', 'communicative', 'transactional',
                        'educational', 'conflict'
                    )
                ),
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Prevent self-loops
                CHECK(source_actor_id != target_actor_id),
                
                -- Prevent exact duplicates
                UNIQUE(source_actor_id, target_actor_id, relationship_type)
            )
        ''')
        
        # Indexes for fast lookup
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_actor_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_actor_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relationship_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edges_timestamp ON edges(created_timestamp)')
        
        self.conn.commit()
        logger.info("All database tables initialized")


# ============================================================================
# LAYER 2: BUSINESS LOGIC - ACTOR MANAGEMENT
# ============================================================================

class ActorManager:
    """Manager for actor identity and biographical operations"""
    
    def __init__(self, db: NetworkDatabase):
        self.db = db

    # Actor ID Tab files
    
    def create_actor_id(self, nationality: str, name: str, born: str, 
                       source: str = None) -> str:
        """Create new immutable actor identity"""
        
        if not nationality or not name or not born:
            raise ValueError("Nationality, name, and birth are required")
        
        actor_id = str(uuid.uuid4())
        
        # Determine which name field to populate
        name_fields = {
            'ru': 'name_ru',
            'zh_prc': 'name_zh_prc',
            'zh_hk': 'name_zh_hk',
            'zh_sg': 'name_zh_sg',
            'zh_tw': 'name_zh_tw',
            'other': 'name_en'
        }
        
        name_field = name_fields.get(nationality, 'name_en')
        
        query = f'''
            INSERT INTO actors_id (actor_id, nationality, {name_field}, born_year_month, source)
            VALUES (?, ?, ?, ?, ?)
        '''
        
        try:
            self.db.execute_insert(query, (actor_id, nationality, name, born, source))
            logger.info(f"Created actor_id: {actor_id} ({name})")
            return actor_id
        except Exception as e:
            logger.error(f"Error creating actor_id: {e}")
            raise
    
    def get_actor_id_info(self, actor_id: str) -> Optional[Dict]:
        """Get immutable actor identity information"""
        query = 'SELECT * FROM actors_id WHERE actor_id = ?'
        results = self.db.execute_query(query, (actor_id,))
        return dict(results[0]) if results else None
    
    def add_actor_record(self, actor_id: str, observation_date: str, 
                        **biographical_data) -> int:
        """Add biographical snapshot for actor"""
        
        fields = ['actor_id', 'observation_date'] + list(biographical_data.keys())
        values = [actor_id, observation_date] + list(biographical_data.values())
        
        placeholders = ', '.join(['?' for _ in fields])
        field_names = ', '.join(fields)
        
        query = f'INSERT INTO Actors ({field_names}) VALUES ({placeholders})'
        
        try:
            record_id = self.db.execute_insert(query, tuple(values))
            logger.info(f"Added actor record: {record_id} for {actor_id}")
            return record_id
        except Exception as e:
            logger.error(f"Error adding actor record: {e}")
            raise
    
    def get_current_actor_status(self, actor_id: str) -> Optional[Dict]:
        """Get current (most recent) biographical status"""
        query = '''
            SELECT * FROM Actors 
            WHERE actor_id = ? AND record_status = 'current'
            ORDER BY observation_date DESC 
            LIMIT 1
        '''
        results = self.db.execute_query(query, (actor_id,))
        return dict(results[0]) if results else None
    
    def get_actor_history(self, actor_id: str) -> List[Dict]:
        """Get full biographical history"""
        query = '''
            SELECT * FROM Actors 
            WHERE actor_id = ? 
            ORDER BY observation_date DESC
        '''
        results = self.db.execute_query(query, (actor_id,))
        return [dict(row) for row in results]
    
    def search_actors(self, search_term: str) -> List[Dict]:
        """Search actors by name or other fields"""
        query = '''
            SELECT DISTINCT a.*, aid.* 
            FROM Actors a
            JOIN actors_id aid ON a.actor_id = aid.actor_id
            WHERE a.prof_position LIKE ? 
               OR a.affiliation LIKE ? 
               OR a.status LIKE ?
               OR aid.name_ru LIKE ?
               OR aid.name_zh_prc LIKE ?
               OR aid.name_en LIKE ?
            LIMIT 50
        '''
        search_pattern = f'%{search_term}%'
        results = self.db.execute_query(query, tuple([search_pattern] * 6))
        return [dict(row) for row in results]
    
    # Actors Biographical Tab files
    def add_actor_alias(self, actor_id: str, alias_name: str, alias_type: str,
                       language_code: str = None, confidence: float = 0.8,
                       source: str = None) -> int:
        """Add name alias for actor"""
        query = '''
            INSERT INTO Actor_Aliases (actor_id, alias_name, alias_type, language_code, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        try:
            alias_id = self.db.execute_insert(query, (actor_id, alias_name, alias_type, language_code, confidence, source))
            logger.info(f"Added alias: {alias_name} for actor {actor_id}")
            return alias_id
        except Exception as e:
            logger.error(f"Error adding alias: {e}")
            raise

    def get_actor_aliases(self, actor_id: str) -> List[Dict]:
        """Get all aliases for an actor"""
        query = 'SELECT * FROM Actor_Aliases WHERE actor_id = ? ORDER BY confidence DESC'
        results = self.db.execute_query(query, (actor_id,))
        return [dict(row) for row in results]

    def delete_actor_alias(self, alias_id: int) -> bool:
        """Delete an actor alias"""
        query = 'DELETE FROM Actor_Aliases WHERE alias_id = ?'
        try:
            self.db.execute_insert(query, (alias_id,))
            logger.info(f"Deleted alias: {alias_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting alias: {e}")
            raise

    def get_actor_records_history(self, actor_id: str) -> List[Dict]:
        """Get all historical records for an actor"""
        query = '''
            SELECT * FROM Actors 
            WHERE actor_id = ? 
            ORDER BY observation_date DESC
        '''
        results = self.db.execute_query(query, (actor_id,))
        return [dict(row) for row in results]

    def update_actor_record_status(self, record_id: int, new_status: str) -> bool:
        """Update record status (current/superseded/rejected)"""
        query = 'UPDATE Actors SET record_status = ?, updated_at = CURRENT_TIMESTAMP WHERE record_id = ?'
        try:
            self.db.execute_insert(query, (new_status, record_id))
            logger.info(f"Updated record {record_id} status to {new_status}")
            return True
        except Exception as e:
            logger.error(f"Error updating record status: {e}")
            raise


# ============================================================================
# PART 2: BUSINESS LOGIC - EVENT MANAGER
# ============================================================================

class EventManager:
    """Manager for event operations"""
    
    # Constants
    DOMAINS = ['politics', 'economics', 'society', 'security', 'technology', 
               'healthcare', 'culture', 'sports', 'nature', 'unknown']
    
    TYPES = ['trigger', 'process', 'institutional', 'normative', 
             'symbol_narrative', 'forecast', 'unknown']
    
    SCALES = ['local', 'national', 'regional', 'global', 'unknown']
    
    PRIORITIES = ['strategic', 'tactical', 'noise', 'unknown']
    
    DATE_PRECISIONS = ['day', 'month', 'year', 'unknown']
    
    def __init__(self, db: 'NetworkDatabase'):
        self.db = db
    
    def validate_event(self, event_data: Dict) -> Tuple[bool, str]:
        """
        Validate event data before insertion
        Returns: (is_valid, error_message)
        """
        
        # Required fields
        if not event_data.get('event_title_ru') or not event_data['event_title_ru'].strip():
            return False, "Error 1: Russian title is required"
        
        if not event_data.get('event_date') or not event_data['event_date'].strip():
            return False, "Error 2: Event date is required"
        
        # Validate date format
        try:
            from datetime import datetime
            datetime.strptime(event_data['event_date'], '%Y-%m-%d')
        except ValueError:
            return False, "Error 2: Event date must be in YYYY-MM-DD format"
        
        if not event_data.get('sources') or not event_data['sources'].strip():
            return False, "Error 3: Sources are required"
        
        # Validate enumerations
        precision = event_data.get('event_date_precision', 'day')
        if precision not in self.DATE_PRECISIONS:
            return False, f"Error 4: Invalid date precision. Must be one of: {', '.join(self.DATE_PRECISIONS)}"
        
        domain = event_data.get('event_domain', 'unknown')
        if domain not in self.DOMAINS:
            return False, f"Error 5: Invalid domain. Must be one of: {', '.join(self.DOMAINS)}"
        
        event_type = event_data.get('event_type', 'unknown')
        if event_type not in self.TYPES:
            return False, f"Error 6: Invalid event type. Must be one of: {', '.join(self.TYPES)}"
        
        scale = event_data.get('event_scale', 'unknown')
        if scale not in self.SCALES:
            return False, f"Error 7: Invalid scale. Must be one of: {', '.join(self.SCALES)}"
        
        priority = event_data.get('event_priority', 'unknown')
        if priority not in self.PRIORITIES:
            return False, f"Error 8: Invalid priority. Must be one of: {', '.join(self.PRIORITIES)}"
        
        # Validate participants_count if provided
        if event_data.get('participants_count'):
            try:
                int(event_data['participants_count'])
            except (ValueError, TypeError):
                return False, "Error 9: Participants count must be an integer"
        
        return True, ""
    
    def create_event(self, event_data: Dict) -> int:
        """Create new event, returns event_id"""
        
        is_valid, error_msg = self.validate_event(event_data)
        if not is_valid:
            raise ValueError(error_msg)
        
        query = '''
            INSERT INTO events (
                event_title_ru, event_title_zh, event_title_en,
                event_date, event_date_precision,
                description_ru, description_zh, description_en,
                event_domain, event_type, event_scale, event_priority,
                participants_count, sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            event_data['event_title_ru'],
            event_data.get('event_title_zh'),
            event_data.get('event_title_en'),
            event_data['event_date'],
            event_data.get('event_date_precision', 'day'),
            event_data.get('description_ru'),
            event_data.get('description_zh'),
            event_data.get('description_en'),
            event_data.get('event_domain', 'unknown'),
            event_data.get('event_type', 'unknown'),
            event_data.get('event_scale', 'unknown'),
            event_data.get('event_priority', 'unknown'),
            event_data.get('participants_count'),
            event_data['sources']
        )
        
        try:
            event_id = self.db.execute_insert(query, params)
            logger.info(f"Created event: {event_id} ({event_data['event_title_ru']})")
            return event_id
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            raise
    
    def update_event(self, event_id: int, event_data: Dict) -> bool:
        """Update existing event"""
        
        is_valid, error_msg = self.validate_event(event_data)
        if not is_valid:
            raise ValueError(error_msg)
        
        query = '''
            UPDATE events SET
                event_title_ru = ?, event_title_zh = ?, event_title_en = ?,
                event_date = ?, event_date_precision = ?,
                description_ru = ?, description_zh = ?, description_en = ?,
                event_domain = ?, event_type = ?, event_scale = ?, event_priority = ?,
                participants_count = ?, sources = ?
            WHERE event_id = ?
        '''
        
        params = (
            event_data['event_title_ru'],
            event_data.get('event_title_zh'),
            event_data.get('event_title_en'),
            event_data['event_date'],
            event_data.get('event_date_precision', 'day'),
            event_data.get('description_ru'),
            event_data.get('description_zh'),
            event_data.get('description_en'),
            event_data.get('event_domain', 'unknown'),
            event_data.get('event_type', 'unknown'),
            event_data.get('event_scale', 'unknown'),
            event_data.get('event_priority', 'unknown'),
            event_data.get('participants_count'),
            event_data['sources'],
            event_id
        )
        
        try:
            self.db.execute_insert(query, params)
            logger.info(f"Updated event: {event_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating event: {e}")
            raise
    
    def get_event(self, event_id: int) -> Optional[Dict]:
        """Get event by ID"""
        query = 'SELECT * FROM events WHERE event_id = ?'
        result = self.db.execute_query(query, (event_id,))
        return dict(result[0]) if result else None
    
    def get_all_events(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all events with pagination"""
        query = 'SELECT * FROM events ORDER BY event_date DESC, event_id DESC LIMIT ? OFFSET ?'
        results = self.db.execute_query(query, (limit, offset))
        return [dict(row) for row in results]
    
    def search_events(self, search_term: str = '', filters: Dict = None, 
                     limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        Search events by quick search and/or advanced filters
        
        filters dict can contain:
        - event_date: date string
        - event_date_precision: 'day', 'month', 'year'
        - event_domain: domain string
        - event_type: type string
        - event_scale: scale string
        - event_priority: priority string
        """
        
        query = 'SELECT * FROM events WHERE 1=1'
        params = []
        
        # Quick search on titles
        if search_term.strip():
            query += ' AND (event_title_ru LIKE ? OR event_title_zh LIKE ? OR event_title_en LIKE ?)'
            pattern = f'%{search_term}%'
            params.extend([pattern, pattern, pattern])
        
        # Advanced filters
        if filters:
            if filters.get('event_date'):
                query += ' AND event_date = ?'
                params.append(filters['event_date'])
            
            if filters.get('event_date_precision'):
                query += ' AND event_date_precision = ?'
                params.append(filters['event_date_precision'])
            
            if filters.get('event_domain') and filters['event_domain'] != 'unknown':
                query += ' AND event_domain = ?'
                params.append(filters['event_domain'])
            
            if filters.get('event_type') and filters['event_type'] != 'unknown':
                query += ' AND event_type = ?'
                params.append(filters['event_type'])
            
            if filters.get('event_scale') and filters['event_scale'] != 'unknown':
                query += ' AND event_scale = ?'
                params.append(filters['event_scale'])
            
            if filters.get('event_priority') and filters['event_priority'] != 'unknown':
                query += ' AND event_priority = ?'
                params.append(filters['event_priority'])
        
        query += ' ORDER BY event_date DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        try:
            results = self.db.execute_query(query, tuple(params))
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_event_count(self) -> int:
        """Get total event count"""
        query = 'SELECT COUNT(*) as cnt FROM events'
        result = self.db.execute_query(query)
        return dict(result[0])['cnt'] if result else 0
    
    def delete_event(self, event_id: int) -> bool:
        """Delete event by ID"""
        query = 'DELETE FROM events WHERE event_id = ?'
        try:
            self.db.execute_insert(query, (event_id,))
            logger.info(f"Deleted event: {event_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting event: {e}")
            raise


class ParticipationManager:
    """Manager for event participation operations"""
    
    # Constants
    ROLES = ['organizer', 'executor', 'mediator', 'beneficiary', 
             'affected_party', 'provocateur', 'observer', 'unknown']
    
    FUNCTIONS = ['leader', 'spokesperson', 'expert', 'witness', 
                 'information_source', 'unknown']
    
    POSITIONS = ['state_actor', 'commercial_actor', 'non_commercial_actor', 
                 'international_actor', 'illegal_actor', 'religious_actor', 
                 'individual_actor', 'unknown']
    
    INFO_ROLES = ['information_spreader', 'information_target', 
                  'narrative_creator', 'unknown']
    
    def __init__(self, db: 'NetworkDatabase'):
        self.db = db
    
    def validate_participation(self, participation_data: Dict) -> Tuple[bool, str]:
        """
        Validate participation data before insertion
        Returns: (is_valid, error_message)
        """
        
        # Required fields
        if not participation_data.get('event_id'):
            return False, "Error 1: Event ID is required"
        
        if not participation_data.get('actor_id') or not participation_data['actor_id'].strip():
            return False, "Error 2: Actor ID is required"
        
        # Verify event exists
        try:
            event_result = self.db.execute_query(
                'SELECT event_id FROM events WHERE event_id = ?',
                (participation_data['event_id'],)
            )
            if not event_result:
                return False, f"Error 1: Event ID {participation_data['event_id']} not found"
        except Exception as e:
            logger.error(f"Error checking event: {e}")
            return False, f"Error 1: Database error checking event: {str(e)}"
        
        # Verify actor exists
        try:
            actor_result = self.db.execute_query(
                'SELECT actor_id FROM actors_id WHERE actor_id = ?',
                (participation_data['actor_id'],)
            )
            if not actor_result:
                return False, f"Error 2: Actor ID not found"
        except Exception as e:
            logger.error(f"Error checking actor: {e}")
            return False, f"Error 2: Database error checking actor: {str(e)}"
        
        # Validate enumerations
        role = participation_data.get('participation_role', 'unknown')
        if role not in self.ROLES:
            return False, f"Error 3: Invalid participation role. Must be one of: {', '.join(self.ROLES)}"
        
        function = participation_data.get('participation_function', 'unknown')
        if function not in self.FUNCTIONS:
            return False, f"Error 4: Invalid participation function. Must be one of: {', '.join(self.FUNCTIONS)}"
        
        position = participation_data.get('participation_position', 'unknown')
        if position not in self.POSITIONS:
            return False, f"Error 5: Invalid participation position. Must be one of: {', '.join(self.POSITIONS)}"
        
        info_role = participation_data.get('information_role', 'unknown')
        if info_role not in self.INFO_ROLES:
            return False, f"Error 6: Invalid information role. Must be one of: {', '.join(self.INFO_ROLES)}"
        
        return True, ""
    
    def create_participation(self, participation_data: Dict) -> int:
        """Create new participation, returns participation_id"""
        
        is_valid, error_msg = self.validate_participation(participation_data)
        if not is_valid:
            raise ValueError(error_msg)
        
        query = '''
            INSERT INTO event_participation (
                event_id, actor_id, participation_role, participation_function,
                participation_position, information_role, description, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            participation_data['event_id'],
            participation_data['actor_id'],
            participation_data.get('participation_role', 'unknown'),
            participation_data.get('participation_function', 'unknown'),
            participation_data.get('participation_position', 'unknown'),
            participation_data.get('information_role', 'unknown'),
            participation_data.get('description'),
            participation_data.get('source')
        )
        
        try:
            participation_id = self.db.execute_insert(query, params)
            logger.info(f"Created participation: {participation_id} (event {participation_data['event_id']}, actor {participation_data['actor_id']})")
            return participation_id
        except sqlite3.IntegrityError as e:
            if 'UNIQUE constraint failed' in str(e):
                raise ValueError("Error 7: This actor already participates in this event")
            raise
        except Exception as e:
            logger.error(f"Error creating participation: {e}")
            raise
    
    def update_participation(self, participation_id: int, participation_data: Dict) -> bool:
        """Update existing participation"""
        
        is_valid, error_msg = self.validate_participation(participation_data)
        if not is_valid:
            raise ValueError(error_msg)
        
        query = '''
            UPDATE event_participation SET
                participation_role = ?, participation_function = ?,
                participation_position = ?, information_role = ?,
                description = ?, source = ?
            WHERE participation_id = ?
        '''
        
        params = (
            participation_data.get('participation_role', 'unknown'),
            participation_data.get('participation_function', 'unknown'),
            participation_data.get('participation_position', 'unknown'),
            participation_data.get('information_role', 'unknown'),
            participation_data.get('description'),
            participation_data.get('source'),
            participation_id
        )
        
        try:
            self.db.execute_insert(query, params)
            logger.info(f"Updated participation: {participation_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating participation: {e}")
            raise
    
    def get_participation(self, participation_id: int) -> Optional[Dict]:
        """Get participation by ID"""
        query = 'SELECT * FROM event_participation WHERE participation_id = ?'
        result = self.db.execute_query(query, (participation_id,))
        return dict(result[0]) if result else None
    
    def get_participations_for_event(self, event_id: int) -> List[Dict]:
        """Get all participations for an event"""
        query = '''
            SELECT ep.*, e.event_title_ru, a.name_ru 
            FROM event_participation ep
            JOIN events e ON ep.event_id = e.event_id
            JOIN actors_id a ON ep.actor_id = a.actor_id
            WHERE ep.event_id = ?
            ORDER BY ep.created_at DESC
        '''
        results = self.db.execute_query(query, (event_id,))
        return [dict(row) for row in results]
    
    def get_participations_for_actor(self, actor_id: str) -> List[Dict]:
        """Get all participations for an actor"""
        query = '''
            SELECT ep.*, e.event_title_ru, a.name_ru 
            FROM event_participation ep
            JOIN events e ON ep.event_id = e.event_id
            JOIN actors_id a ON ep.actor_id = a.actor_id
            WHERE ep.actor_id = ?
            ORDER BY ep.created_at DESC
        '''
        results = self.db.execute_query(query, (actor_id,))
        return [dict(row) for row in results]
    
    def get_all_participations(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all participations with pagination"""
        query = '''
            SELECT ep.*, e.event_title_ru, a.name_ru 
            FROM event_participation ep
            JOIN events e ON ep.event_id = e.event_id
            JOIN actors_id a ON ep.actor_id = a.actor_id
            ORDER BY ep.created_at DESC LIMIT ? OFFSET ?
        '''
        results = self.db.execute_query(query, (limit, offset))
        return [dict(row) for row in results]
    
    def delete_participation(self, participation_id: int) -> bool:
        """Delete participation by ID"""
        query = 'DELETE FROM event_participation WHERE participation_id = ?'
        try:
            self.db.execute_insert(query, (participation_id,))
            logger.info(f"Deleted participation: {participation_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting participation: {e}")
            raise
    
    def get_participation_count(self) -> int:
        """Get total participation count"""
        query = 'SELECT COUNT(*) as cnt FROM event_participation'
        result = self.db.execute_query(query)
        return dict(result[0])['cnt'] if result else 0


class SearchHistoryManager:
    """Manages persistent search history in JSON"""
    
    def __init__(self, history_file: str = 'search_event_history.json'):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load history from JSON file"""
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Error loading search history: {e}")
            return []
    
    def _save_history(self):
        """Save history to JSON file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving search history: {e}")
    
    def add_search(self, event_data: Dict):
        """Add event to search history (most recent first)"""
        entry = {
            'event_id': event_data.get('event_id'),
            'event_date': event_data.get('event_date'),
            'event_title_ru': event_data.get('event_title_ru'),
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.insert(0, entry)
        
        # Keep only last 21 entries
        self.history = self.history[:21]
        self._save_history()
    
    def get_history(self) -> List[Dict]:
        """Get search history (most recent first)"""
        return self.history
    
    def clear_history(self):
        """Clear all history"""
        self.history = []
        self._save_history()


# ============================================================================
# PART 2: BUSINESS LOGIC - EDGES MANAGER
# ============================================================================

class EdgeManager:
    """Manager for relationship/edge operations"""

    # Valid relationship types
    RELATIONSHIP_TYPES = [
        'kinship', 'trust', 'community', 'informal',
        'official', 'mentoring', 'communicative', 'transactional',
        'educational', 'conflict'
    ]

    def __init__(self, db: 'NetworkDatabase'):
        self.db = db

    def validate_edge(self, source_actor_id: str, target_actor_id: str,
                     relationship_type: str) -> tuple[bool, str]:
        """
        Validate edge creation parameters in sequence
        Returns: (is_valid, error_message)
        """

        # Check 1: Verify source_actor_id is populated and valid
        if not source_actor_id or not source_actor_id.strip():
            return False, "Error 1: Source actor ID is empty"

        try:
            query = 'SELECT actor_id FROM actors_id WHERE actor_id = ?'
            result = self.db.execute_query(query, (source_actor_id,))
            if not result:
                return False, f"Error 1: Source actor ID '{source_actor_id}' not found in database"
        except Exception as e:
            logger.error(f"Error validating source: {e}")
            return False, f"Error 1: Database error checking source actor: {str(e)}"

        # Check 2: Verify target_actor_id is populated and valid
        if not target_actor_id or not target_actor_id.strip():
            return False, "Error 2: Target actor ID is empty"

        try:
            result = self.db.execute_query(query, (target_actor_id,))
            if not result:
                return False, f"Error 2: Target actor ID '{target_actor_id}' not found in database"
        except Exception as e:
            logger.error(f"Error validating target: {e}")
            return False, f"Error 2: Database error checking target actor: {str(e)}"

        # Check 3: Verify relationship_type is selected
        if not relationship_type or not relationship_type.strip():
            return False, "Error 3: Relationship type is not selected"

        if relationship_type not in self.RELATIONSHIP_TYPES:
            return False, f"Error 3: Invalid relationship type '{relationship_type}'. Must be one of: {', '.join(self.RELATIONSHIP_TYPES)}"

        # Check 4: Prevent self-loops
        if source_actor_id == target_actor_id:
            return False, "Error 4: Source and target actor cannot be the same (self-loops not allowed)"

        # Check 5 (optional): Check for duplicate edges
        try:
            dup_query = '''
                SELECT edge_id FROM edges
                WHERE source_actor_id = ? AND target_actor_id = ? AND relationship_type = ?
            '''
            dup_result = self.db.execute_query(dup_query, (source_actor_id, target_actor_id, relationship_type))
            if dup_result:
                return False, f"Error 5 (Warning): Duplicate edge already exists. Edge ID: {dict(dup_result[0])['edge_id']}"
        except Exception as e:
            logger.warning(f"Could not check for duplicates: {e}")
            # Don't fail on this check, just warn

        return True, ""

    def create_edge(self, source_actor_id: str, target_actor_id: str,
                   relationship_type: str) -> int:
        """Create new edge/relationship, returns edge_id"""

        # Validate first
        is_valid, error_msg = self.validate_edge(source_actor_id, target_actor_id, relationship_type)
        if not is_valid:
            raise ValueError(error_msg)

        query = '''
            INSERT INTO edges (source_actor_id, target_actor_id, relationship_type)
            VALUES (?, ?, ?)
        '''

        try:
            edge_id = self.db.execute_insert(query, (source_actor_id, target_actor_id, relationship_type))
            logger.info(f"Created edge: {edge_id} ({source_actor_id} -> {target_actor_id}, type: {relationship_type})")
            return edge_id
        except Exception as e:
            logger.error(f"Error creating edge: {e}")
            raise

    def get_edge(self, edge_id: int) -> dict | None:
        """Get edge by ID"""
        query = 'SELECT * FROM edges WHERE edge_id = ?'
        result = self.db.execute_query(query, (edge_id,))
        return dict(result[0]) if result else None

    def get_edges_for_actor(self, actor_id: str, direction: str = 'both') -> list[dict]:
        """Get all edges for an actor

        direction: 'outgoing' (source), 'incoming' (target), 'both'
        """
        if direction == 'outgoing':
            query = 'SELECT * FROM edges WHERE source_actor_id = ? ORDER BY created_timestamp DESC'
            results = self.db.execute_query(query, (actor_id,))
        elif direction == 'incoming':
            query = 'SELECT * FROM edges WHERE target_actor_id = ? ORDER BY created_timestamp DESC'
            results = self.db.execute_query(query, (actor_id,))
        else:  # both
            query = '''
                SELECT * FROM edges
                WHERE source_actor_id = ? OR target_actor_id = ?
                ORDER BY created_timestamp DESC
            '''
            results = self.db.execute_query(query, (actor_id, actor_id))

        return [dict(row) for row in results]

    def get_all_edges(self, relationship_type: str = None, limit: int = None,
                     offset: int = 0) -> list[dict]:
        """Get all edges with optional filtering and pagination"""

        if relationship_type:
            query = 'SELECT * FROM edges WHERE relationship_type = ?'
            params = (relationship_type,)
        else:
            query = 'SELECT * FROM edges'
            params = ()

        query += ' ORDER BY created_timestamp DESC'

        if limit:
            query += f' LIMIT {limit} OFFSET {offset}'

        results = self.db.execute_query(query, params)
        return [dict(row) for row in results]

    def delete_edge(self, edge_id: int) -> bool:
        """Delete edge by ID"""
        query = 'DELETE FROM edges WHERE edge_id = ?'
        try:
            self.db.execute_insert(query, (edge_id,))
            logger.info(f"Deleted edge: {edge_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting edge: {e}")
            raise

    def get_edge_count(self, relationship_type: str = None) -> int:
        """Get count of edges, optionally filtered by type"""
        if relationship_type:
            query = 'SELECT COUNT(*) as cnt FROM edges WHERE relationship_type = ?'
            result = self.db.execute_query(query, (relationship_type,))
        else:
            query = 'SELECT COUNT(*) as cnt FROM edges'
            result = self.db.execute_query(query)

        return dict(result[0])['cnt'] if result else 0


# ============================================================================
# LAYER 2: BUSINESS LOGIC - NAME LOOKUP (5-PHASE ALGORITHM)
# ============================================================================

class NameLookupEngine:
    """Intelligent multilingual name lookup with 5-phase algorithm"""
    
    LEVENSHTEIN_EXACT_THRESHOLD = 0.95
    LEVENSHTEIN_FUZZY_THRESHOLD = 0.85
    
    def __init__(self, db: NetworkDatabase):
        self.db = db
    
    def find_actor_by_name(self, name: str, language: str = None) -> LookupResult:
        """Main entry point: execute 5-phase lookup"""
        
        logger.info(f"Looking up name: {name} (language: {language})")
        
        # Phase 1: Exact match
        result = self._phase_exact_match(name, language)
        if result.status == 'found':
            result.confidence = 1.0
            return result
        
        # Phase 2: Fuzzy match
        result = self._phase_fuzzy_match(name, language)
        if result.status == 'found':
            return result
        
        # Phase 3: Alias lookup
        result = self._phase_alias_lookup(name)
        if result.status == 'found':
            return result
        
        # Phase 4: Pinyin matching (Chinese)
        if language and 'zh' in language:
            result = self._phase_pinyin_match(name)
            if result.status == 'found':
                return result
        
        # Phase 5: Disambiguation or error
        return self._phase_disambiguation(name, language)
    
    def _phase_exact_match(self, name: str, language: str = None) -> LookupResult:
        """Phase 1: Exact name match in actors_id"""
        
        name_fields = {
            'ru': 'name_ru',
            'zh_prc': 'name_zh_prc',
            'zh_hk': 'name_zh_hk',
            'zh_sg': 'name_zh_sg',
            'zh_tw': 'name_zh_tw',
            'en': 'name_en',
            'other': 'name_en'
        }
        
        if language and language in name_fields:
            field = name_fields[language]
            query = f'SELECT * FROM actors_id WHERE {field} = ?'
            results = self.db.execute_query(query, (name,))
            
            if len(results) == 1:
                row = dict(results[0])
                logger.info(f"Phase 1 exact match found: {row['actor_id']}")
                return LookupResult(
                    status='found',
                    actor_id=row['actor_id'],
                    confidence=1.0,
                    message=f"Exact match: {name}"
                )
            elif len(results) > 1:
                candidates = [dict(r) for r in results]
                return LookupResult(
                    status='ambiguous',
                    candidates=candidates,
                    message=f"Multiple exact matches found"
                )
        else:
            # Search all name fields
            query = '''
                SELECT * FROM actors_id 
                WHERE name_ru = ? OR name_zh_prc = ? OR name_zh_hk = ? 
                   OR name_zh_sg = ? OR name_zh_tw = ? OR name_en = ?
            '''
            results = self.db.execute_query(query, tuple([name] * 6))
            
            if len(results) == 1:
                row = dict(results[0])
                return LookupResult(
                    status='found',
                    actor_id=row['actor_id'],
                    confidence=1.0,
                    message=f"Exact match: {name}"
                )
            elif len(results) > 1:
                candidates = [dict(r) for r in results]
                return LookupResult(
                    status='ambiguous',
                    candidates=candidates,
                    message=f"Multiple exact matches found"
                )
        
        return LookupResult(status='not_found')
    
    def _phase_fuzzy_match(self, name: str, language: str = None) -> LookupResult:
        """Phase 2: Fuzzy match using Levenshtein distance"""
        
        query = 'SELECT * FROM actors_id'
        all_actors = self.db.execute_query(query)
        
        candidates = []
        
        for actor in all_actors:
            actor_dict = dict(actor)
            
            # Check all name fields
            for field_name in ['name_ru', 'name_zh_prc', 'name_zh_hk', 'name_zh_sg', 'name_zh_tw', 'name_en']:
                if actor_dict[field_name]:
                    similarity = difflib.SequenceMatcher(None, name, actor_dict[field_name]).ratio()
                    
                    if similarity >= self.LEVENSHTEIN_FUZZY_THRESHOLD:
                        candidates.append({
                            'actor_id': actor_dict['actor_id'],
                            'matched_name': actor_dict[field_name],
                            'confidence': similarity,
                            'source': 'fuzzy_match'
                        })
        
        if len(candidates) == 1 and candidates[0]['confidence'] >= self.LEVENSHTEIN_EXACT_THRESHOLD:
            logger.info(f"Phase 2 high-confidence fuzzy match: {candidates[0]['actor_id']}")
            return LookupResult(
                status='found',
                actor_id=candidates[0]['actor_id'],
                confidence=candidates[0]['confidence'],
                message=f"High-confidence fuzzy match: {candidates[0]['matched_name']}"
            )
        elif len(candidates) > 1:
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            return LookupResult(
                status='ambiguous',
                candidates=candidates,
                message=f"Found {len(candidates)} fuzzy matches"
            )
        
        return LookupResult(status='not_found')
    
    def _phase_alias_lookup(self, name: str) -> LookupResult:
        """Phase 3: Lookup in Actor_Aliases table"""
        
        query = 'SELECT DISTINCT actor_id, confidence FROM Actor_Aliases WHERE alias_name = ?'
        results = self.db.execute_query(query, (name,))
        
        if len(results) == 1:
            result = dict(results[0])
            logger.info(f"Phase 3 alias match: {result['actor_id']}")
            return LookupResult(
                status='found',
                actor_id=result['actor_id'],
                confidence=result['confidence'],
                message=f"Alias match: {name}"
            )
        elif len(results) > 1:
            candidates = [{'actor_id': dict(r)['actor_id'], 'confidence': dict(r)['confidence']} 
                         for r in results]
            return LookupResult(
                status='ambiguous',
                candidates=candidates,
                message=f"Multiple alias matches found"
            )
        
        return LookupResult(status='not_found')
    
    def _phase_pinyin_match(self, name: str) -> LookupResult:
        """Phase 4: Pinyin matching for Chinese names (placeholder)"""
        # Note: Full pinyin implementation would require pypinyin library
        # This is a simplified version
        
        query = 'SELECT * FROM actors_id WHERE pinyin LIKE ?'
        pattern = f'%{name}%'
        results = self.db.execute_query(query, (pattern,))
        
        if len(results) >= 1:
            candidates = [dict(r) for r in results]
            logger.info(f"Phase 4 pinyin matches: {len(candidates)}")
            return LookupResult(
                status='ambiguous' if len(results) > 1 else 'found',
                actor_id=candidates[0]['actor_id'] if len(results) == 1 else None,
                candidates=candidates,
                confidence=0.8,
                message=f"Pinyin match (confidence: 80%)"
            )
        
        return LookupResult(status='not_found')
    
    def _phase_disambiguation(self, name: str, language: str = None) -> LookupResult:
        """Phase 5: Handle disambiguation or not found"""
        
        logger.warning(f"Phase 5: No match found for '{name}'")
        return LookupResult(
            status='not_found',
            message=f"No actor found with name: {name}\nOptions: Create new actor or enter manually"
        )


# ============================================================================
# LAYER 3: GUI - MAIN APPLICATION
# ============================================================================

class SearchEventsModal(tk.Toplevel):
    """Modal dialog for searching and selecting events"""
    
    def __init__(self, parent, event_manager: EventManager, 
                 history_manager: SearchHistoryManager, title: str = "Search Events"):
        super().__init__(parent)
        self.title(title)
        self.geometry("1000x700")
        self.event_manager = event_manager
        self.history_manager = history_manager
        
        self.selected_event = None
        self.search_after_id = None
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_ui()
        self.focus()
    
    def create_ui(self):
        """Create modal UI"""
        
        # ===== SEARCH FRAME =====
        search_frame = ttk.LabelFrame(self, text="🔍 Quick Search", padding=10)
        search_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.quick_search_var = tk.StringVar()
        self.quick_search_entry = ttk.Entry(search_frame, textvariable=self.quick_search_var, 
                                            width=60)
        self.quick_search_entry.pack(fill=tk.X)
        self.quick_search_entry.insert(0, "Search by title (RU/ZH/EN)...")
        
        # Bind with debounce
        self.quick_search_var.trace('w', self.on_quick_search_changed)
        
        # ===== ADVANCED SEARCH FRAME =====
        adv_frame = ttk.LabelFrame(self, text="⚙️ Advanced Search", padding=10)
        adv_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Date row
        ttk.Label(adv_frame, text="Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W)
        self.date_var = tk.StringVar()
        ttk.Entry(adv_frame, textvariable=self.date_var, width=15).grid(row=0, column=1, padx=5)
        
        # Date precision dropdown
        ttk.Label(adv_frame, text="Precision:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.precision_var = tk.StringVar(value="")
        ttk.Combobox(adv_frame, textvariable=self.precision_var, 
                    values=['', 'day', 'month', 'year'], state='readonly', width=12).grid(row=0, column=3, padx=5)
        
        # Domain
        ttk.Label(adv_frame, text="Domain:").grid(row=1, column=0, sticky=tk.W)
        self.domain_var = tk.StringVar(value="")
        ttk.Combobox(adv_frame, textvariable=self.domain_var, 
                    values=[''] + self.event_manager.DOMAINS, state='readonly', width=15).grid(row=1, column=1, padx=5)
        
        # Type
        ttk.Label(adv_frame, text="Type:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0))
        self.type_var = tk.StringVar(value="")
        ttk.Combobox(adv_frame, textvariable=self.type_var, 
                    values=[''] + self.event_manager.TYPES, state='readonly', width=12).grid(row=1, column=3, padx=5)
        
        # Scale
        ttk.Label(adv_frame, text="Scale:").grid(row=2, column=0, sticky=tk.W)
        self.scale_var = tk.StringVar(value="")
        ttk.Combobox(adv_frame, textvariable=self.scale_var, 
                    values=[''] + self.event_manager.SCALES, state='readonly', width=15).grid(row=2, column=1, padx=5)
        
        # Priority
        ttk.Label(adv_frame, text="Priority:").grid(row=2, column=2, sticky=tk.W, padx=(20, 0))
        self.priority_var = tk.StringVar(value="")
        ttk.Combobox(adv_frame, textvariable=self.priority_var, 
                    values=[''] + self.event_manager.PRIORITIES, state='readonly', width=12).grid(row=2, column=3, padx=5)
        
        # Reset button
        ttk.Button(adv_frame, text="🔄 Reset Filters", 
                  command=self.reset_filters).grid(row=3, column=0, sticky=tk.W, pady=10)
        
        # ===== HISTORY FRAME =====
        hist_frame = ttk.LabelFrame(self, text="📋 Search History (Last 21)", padding=10)
        hist_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.history_listbox = tk.Listbox(hist_frame, height=4)
        self.history_listbox.pack(fill=tk.X)
        self.history_listbox.bind('<<ListboxSelect>>', self.on_history_selected)
        
        self.refresh_history()
        
        # ===== RESULTS FRAME =====
        results_frame = ttk.LabelFrame(self, text="📊 Search Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_tree = ttk.Treeview(results_frame, 
                                        columns=('ID', 'Date', 'Title (RU)'),
                                        height=15, show='tree headings')
        
        self.results_tree.column('#0', width=0, stretch=tk.NO)
        self.results_tree.column('ID', anchor=tk.CENTER, width=50)
        self.results_tree.column('Date', anchor=tk.CENTER, width=100)
        self.results_tree.column('Title (RU)', anchor=tk.W, width=700)
        
        self.results_tree.heading('ID', text='ID')
        self.results_tree.heading('Date', text='Date')
        self.results_tree.heading('Title (RU)', text='Title (RU)')
        
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', 
                                 command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_tree.bind('<Double-1>', lambda e: self.confirm_selection())
        
        # ===== ACTION BUTTONS =====
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="✓ Select", 
                  command=self.confirm_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="✕ Cancel", 
                  command=self.cancel).pack(side=tk.RIGHT, padx=5)
        
        # Initial search
        self.execute_search()
    
    def on_quick_search_changed(self, *args):
        """Debounced quick search"""
        if self.search_after_id:
            self.after_cancel(self.search_after_id)
        self.search_after_id = self.after(300, self.execute_search)
    
    def execute_search(self):
        """Execute search with current filters"""
        search_term = self.quick_search_var.get().strip()
        if search_term == "Search by title (RU/ZH/EN)...":
            search_term = ""
        
        filters = {}
        if self.date_var.get().strip():
            filters['event_date'] = self.date_var.get().strip()
        if self.precision_var.get():
            filters['event_date_precision'] = self.precision_var.get()
        if self.domain_var.get():
            filters['event_domain'] = self.domain_var.get()
        if self.type_var.get():
            filters['event_type'] = self.type_var.get()
        if self.scale_var.get():
            filters['event_scale'] = self.scale_var.get()
        if self.priority_var.get():
            filters['event_priority'] = self.priority_var.get()
        
        # Clear results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        try:
            results = self.event_manager.search_events(search_term, filters if filters else None)
            
            for i, event in enumerate(results[:50]):
                self.results_tree.insert('', 'end', iid=f"event_{i}",
                                        values=(event['event_id'], 
                                               event['event_date'],
                                               event['event_title_ru']))
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            messagebox.showerror("Search Error", str(e))
    
    def reset_filters(self):
        """Reset all filters"""
        self.date_var.set("")
        self.precision_var.set("")
        self.domain_var.set("")
        self.type_var.set("")
        self.scale_var.set("")
        self.priority_var.set("")
        self.execute_search()
    
    def refresh_history(self):
        """Refresh history display"""
        self.history_listbox.delete(0, tk.END)
        for entry in self.history_manager.get_history():
            label = f"{entry['event_date']} - {entry['event_title_ru'][:60]}"
            self.history_listbox.insert(tk.END, label)
    
    def on_history_selected(self, event):
        """Handle history selection"""
        selection = self.history_listbox.curselection()
        if not selection:
            return
        
        history = self.history_manager.get_history()
        selected_entry = history[selection[0]]
        
        event_id = selected_entry['event_id']
        event = self.event_manager.get_event(event_id)
        
        if event:
            self.selected_event = dict(event)
            self.confirm_selection()
    
    def confirm_selection(self):
        """Get selected event and close"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an event")
            return
        
        item_id = selection[0]
        values = self.results_tree.item(item_id, 'values')
        
        event_id = int(values[0])
        event = self.event_manager.get_event(event_id)
        
        if event:
            self.selected_event = dict(event)
            self.history_manager.add_search(self.selected_event)
        
        self.destroy()
    
    def cancel(self):
        """Cancel and close"""
        self.selected_event = None
        self.destroy()


# Modal Search Event Class for Participation

class ParticipationSearchEventsModal(tk.Toplevel):
    """Modal dialog for searching and selecting events"""
    
    def __init__(self, parent, db: 'NetworkDatabase', title: str = "Select Event"):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x600")
        self.db = db
        
        self.selected_event = None
        self.search_after_id = None
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_ui()
        self.focus()
    
    def create_ui(self):
        """Create modal UI"""
        
        # Search frame
        search_frame = ttk.LabelFrame(self, text="🔍 Quick Search", padding=10)
        search_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.quick_search_var = tk.StringVar()
        self.quick_search_entry = ttk.Entry(search_frame, textvariable=self.quick_search_var, width=60)
        self.quick_search_entry.pack(fill=tk.X)
        self.quick_search_entry.insert(0, "Search by title (RU/ZH/EN)...")
        
        # Bind with debounce
        self.quick_search_var.trace('w', self.on_quick_search_changed)
        
        # Results frame
        results_frame = ttk.LabelFrame(self, text="📊 Events", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_tree = ttk.Treeview(results_frame, 
                                        columns=('ID', 'Date', 'Title (RU)'),
                                        height=15, show='tree headings')
        
        self.results_tree.column('#0', width=0, stretch=tk.NO)
        self.results_tree.column('ID', anchor=tk.CENTER, width=50)
        self.results_tree.column('Date', anchor=tk.CENTER, width=100)
        self.results_tree.column('Title (RU)', anchor=tk.W, width=650)
        
        self.results_tree.heading('ID', text='ID')
        self.results_tree.heading('Date', text='Date')
        self.results_tree.heading('Title (RU)', text='Title (RU)')
        
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', 
                                 command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_tree.bind('<Double-1>', lambda e: self.confirm_selection())
        
        # Action buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="✓ Select", 
                  command=self.confirm_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="✕ Cancel", 
                  command=self.cancel).pack(side=tk.RIGHT, padx=5)
        
        # Initial search
        self.execute_search()
    
    def on_quick_search_changed(self, *args):
        """Debounced quick search"""
        if self.search_after_id:
            self.after_cancel(self.search_after_id)
        self.search_after_id = self.after(300, self.execute_search)
    
    def execute_search(self):
        """Execute search"""
        search_term = self.quick_search_var.get().strip()
        if search_term == "Search by title (RU/ZH/EN)..." or not search_term:
            search_term = ""
        
        # Clear results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        try:
            if search_term:
                query = '''
                    SELECT event_id, event_date, event_title_ru FROM events
                    WHERE event_title_ru LIKE ? OR event_title_zh LIKE ? OR event_title_en LIKE ?
                    ORDER BY event_date DESC LIMIT 50
                '''
                pattern = f'%{search_term}%'
                results = self.db.execute_query(query, (pattern, pattern, pattern))
            else:
                query = '''
                    SELECT event_id, event_date, event_title_ru FROM events
                    ORDER BY event_date DESC LIMIT 50
                '''
                results = self.db.execute_query(query)
            
            for i, row in enumerate(results):
                event = dict(row)
                self.results_tree.insert('', 'end', iid=f"event_{i}",
                                        values=(event['event_id'], 
                                               event['event_date'],
                                               event['event_title_ru']))
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            messagebox.showerror("Search Error", str(e))
    
    def confirm_selection(self):
        """Get selected event and close"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an event")
            return
        
        item_id = selection[0]
        values = self.results_tree.item(item_id, 'values')
        
        event_id = int(values[0])
        self.selected_event = {'event_id': event_id, 'event_date': values[1], 'event_title_ru': values[2]}
        
        self.destroy()
    
    def cancel(self):
        """Cancel and close"""
        self.selected_event = None
        self.destroy()


# Modal Search Actor Class for Participation
class SearchActorsModal(tk.Toplevel):
    """Modal dialog for searching and selecting actors"""
    
    def __init__(self, parent, db: 'NetworkDatabase', title: str = "Select Actor"):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x600")
        self.db = db
        
        self.selected_actor = None
        self.search_after_id = None
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_ui()
        self.focus()
    
    def create_ui(self):
        """Create modal UI"""
        
        # Search frame
        search_frame = ttk.LabelFrame(self, text="🔍 Quick Search", padding=10)
        search_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.quick_search_var = tk.StringVar()
        self.quick_search_entry = ttk.Entry(search_frame, textvariable=self.quick_search_var, width=60)
        self.quick_search_entry.pack(fill=tk.X)
        self.quick_search_entry.insert(0, "Search by name (RU/ZH/EN/Pinyin)...")
        
        # Bind with debounce
        self.quick_search_var.trace('w', self.on_quick_search_changed)
        
        # Results frame
        results_frame = ttk.LabelFrame(self, text="👤 Actors", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_tree = ttk.Treeview(results_frame, 
                                        columns=('ID', 'Name'),
                                        height=15, show='tree headings')
        
        self.results_tree.column('#0', width=0, stretch=tk.NO)
        self.results_tree.column('ID', anchor=tk.CENTER, width=300)
        self.results_tree.column('Name', anchor=tk.W, width=500)
        
        self.results_tree.heading('ID', text='Actor ID')
        self.results_tree.heading('Name', text='Name')
        
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', 
                                 command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_tree.bind('<Double-1>', lambda e: self.confirm_selection())
        
        # Action buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="✓ Select", 
                  command=self.confirm_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="✕ Cancel", 
                  command=self.cancel).pack(side=tk.RIGHT, padx=5)
        
        # Initial search
        self.execute_search()
    
    def on_quick_search_changed(self, *args):
        """Debounced quick search"""
        if self.search_after_id:
            self.after_cancel(self.search_after_id)
        self.search_after_id = self.after(300, self.execute_search)
    
    def execute_search(self):
        """Execute search"""
        search_term = self.quick_search_var.get().strip()
        if search_term == "Search by name (RU/ZH/EN/Pinyin)..." or not search_term:
            search_term = ""
        
        # Clear results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        try:
            if search_term:
                query = '''
                    SELECT actor_id, name_ru, name_zh_prc, name_en, pinyin FROM actors_id
                    WHERE name_ru LIKE ? OR name_zh_prc LIKE ? OR name_en LIKE ? OR pinyin LIKE ?
                    ORDER BY actor_id LIMIT 50
                '''
                pattern = f'%{search_term}%'
                results = self.db.execute_query(query, (pattern, pattern, pattern, pattern))
            else:
                query = '''
                    SELECT actor_id, name_ru, name_zh_prc, name_en, pinyin FROM actors_id
                    ORDER BY actor_id LIMIT 50
                '''
                results = self.db.execute_query(query)
            
            for i, row in enumerate(results):
                actor = dict(row)
                name = (actor.get('name_ru') or actor.get('name_zh_prc') or 
                       actor.get('name_en') or 'Unknown')
                self.results_tree.insert('', 'end', iid=f"actor_{i}",
                                        values=(actor['actor_id'], name))
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            messagebox.showerror("Search Error", str(e))
    
    def confirm_selection(self):
        """Get selected actor and close"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an actor")
            return
        
        item_id = selection[0]
        values = self.results_tree.item(item_id, 'values')
        
        actor_id = values[0]
        actor_name = values[1]
        self.selected_actor = {'actor_id': actor_id, 'actor_name': actor_name}
        
        self.destroy()
    
    def cancel(self):
        """Cancel and close"""
        self.selected_actor = None
        self.destroy()


class SelectActorModal(tk.Toplevel):
    """Modal dialog for selecting actor by name or UUID"""

    def __init__(self, parent, name_lookup: 'NameLookupEngine', actor_manager: 'ActorManager',
                 db: 'NetworkDatabase', title: str = "Select Actor"):
        super().__init__(parent)
        self.title(title)
        self.geometry("800x600")
        self.name_lookup = name_lookup
        self.actor_manager = actor_manager
        self.db = db
        self.selected_actor_id = None
        self.selected_actor_name = None

        # Make modal
        self.transient(parent)
        self.grab_set()

        self.create_ui()
        self.focus()

    def create_ui(self):
        """Create modal UI"""

        # Search frame
        search_frame = ttk.LabelFrame(self, text="🔍 Search for Actor", padding=10)
        search_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(search_frame, text="Enter actor name or UUID:").pack()

        input_frame = ttk.Frame(search_frame)
        input_frame.pack(fill=tk.X, pady=10)

        self.search_entry = ttk.Entry(input_frame, width=60)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_entry.bind('<Return>', lambda e: self.execute_search())

        ttk.Button(input_frame, text="Search", command=self.execute_search).pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(self, text="📋 Search Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Results tree
        self.results_tree = ttk.Treeview(results_frame, columns=('ID', 'Name', 'Aliases', 'Status'),
                                        height=15, show='tree headings')

        self.results_tree.column('#0', width=0, stretch=tk.NO)
        self.results_tree.column('ID', anchor=tk.W, width=280)
        self.results_tree.column('Name', anchor=tk.W, width=200)
        self.results_tree.column('Aliases', anchor=tk.W, width=150)
        self.results_tree.column('Status', anchor=tk.W, width=80)

        self.results_tree.heading('ID', text='Actor ID (UUID)')
        self.results_tree.heading('Name', text='Name')
        self.results_tree.heading('Aliases', text='Aliases')
        self.results_tree.heading('Status', text='Status')

        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection
        self.results_tree.bind('<<TreeviewSelect>>', self.on_result_selected)

        # Info frame
        self.info_frame = ttk.LabelFrame(self, text="ℹ️ Selected Actor Info", padding=10)
        self.info_frame.pack(fill=tk.X, padx=10, pady=10)

        self.info_label = ttk.Label(self.info_frame, text="No actor selected")
        self.info_label.pack()

        # Action buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="✓ Select", command=self.confirm_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="➕ Create New Actor", command=self.create_new_actor).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="✕ Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=5)

    def execute_search(self):
        """Execute name lookup"""
        search_term = self.search_entry.get().strip()

        if not search_term:
            messagebox.showwarning("Empty Search", "Please enter a name or UUID")
            return

        # Clear results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        try:
            # Try UUID lookup first
            if len(search_term) == 36 and search_term.count('-') == 4:
                actor_info = self.actor_manager.get_actor_id_info(search_term)
                if actor_info:
                    self.display_results([actor_info])
                    return

            # Try name lookup
            result = self.name_lookup.find_actor_by_name(search_term)

            if result.status == 'found':
                actor_info = self.actor_manager.get_actor_id_info(result.actor_id)
                self.display_results([actor_info])

            elif result.status == 'ambiguous':
                # Display candidates
                candidates = result.candidates
                self.display_results(candidates)

            else:  # not_found
                messagebox.showinfo("Not Found", f"No actor found: {result.message}")

        except Exception as e:
            logger.error(f"Search error: {e}")
            messagebox.showerror("Search Error", str(e))

    def display_results(self, results: list[dict]):
        """Display search results in tree"""
        for i, actor in enumerate(results[:50]):  # Limit to 50 results
            actor_id = actor.get('actor_id', '')

            # Get display name
            name_fields = ['name_zh_prc', 'name_ru', 'name_zh_hk', 'name_en']
            name = next((actor.get(f) for f in name_fields if actor.get(f)), "Unknown")

            # Get aliases
            aliases_result = self.db.execute_query(
                'SELECT COUNT(*) as cnt FROM Actor_Aliases WHERE actor_id = ?',
                (actor_id,)
            )
            alias_count = dict(aliases_result[0])['cnt'] if aliases_result else 0

            status = actor.get('verification_status', 'unknown')

            self.results_tree.insert('', 'end', iid=f"result_{i}",
                                    values=(actor_id, name, f"{alias_count} aliases", status))

    def on_result_selected(self, event):
        """Handle result selection"""
        selection = self.results_tree.selection()
        if not selection:
            return

        item_id = selection[0]
        values = self.results_tree.item(item_id, 'values')

        self.selected_actor_id = values[0]
        self.selected_actor_name = values[1]

        # Show actor info
        info_text = f"Selected: {self.selected_actor_name} (ID: {self.selected_actor_id})"
        self.info_label.config(text=info_text)

    def confirm_selection(self):
        """Confirm and close modal"""
        if not self.selected_actor_id:
            messagebox.showwarning("No Selection", "Please select an actor from results")
            return

        self.destroy()

    def create_new_actor(self):
        """Open create new actor dialog"""
        # This would open CreateActorIDDialog
        messagebox.showinfo("Create New", "Switch to 'Create New Actor ID' tab to add new actor")

    def cancel(self):
        """Cancel and close modal"""
        self.selected_actor_id = None
        self.selected_actor_name = None
        self.destroy()

class ActorsBiographicalTab:
    """Complete implementation of 'Actors Biographical' tab"""
    
    # Validation rules
    STATUS_CHOICES = ['active', 'inactive', 'retired', 'expelled', 'missing', 'died']
    RECORD_STATUS_CHOICES = ['current', 'superseded', 'rejected']
    
    def __init__(self, parent_frame, actor_manager, session, db):
        """Initialize Actors Biographical tab"""
        self.actor_manager = actor_manager
        self.session = session
        self.db = db
        self.parent = parent_frame
        
        # Track loaded data
        self.current_record_id = None
        self.current_aliases = []
        
        self.create_ui()
        logger.info("Actors Biographical tab initialized")
    
    def create_ui(self):
        """Create complete UI for biographical tab"""
        
        # Main container with scrollbar
        main_container = ttk.Frame(self.parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Create canvas for scrolling
        self.canvas = tk.Canvas(main_container, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
                
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")  

        self.canvas.configure(yscrollcommand=scrollbar.set)


        def _on_mousewheel(event):
            # Find the widget under the mouse cursor
            widget = event.widget
            scrollable_widget = None
            
            # We go up the widget hierarchy to find the first scrollable one.
            while widget:
                # Check if the widget has scrolling methods
                if hasattr(widget, 'yview') and hasattr(widget, 'yview_scroll'):
                    scrollable_widget = widget
                    break
                widget = widget.master
            
            # If we find a scrollable widget, we perform the scrolling
            if scrollable_widget:
                if event.delta:
                    # Windows/Mac
                    scrollable_widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
                elif event.num == 4:
                    # Linux - wheel up
                    scrollable_widget.yview_scroll(-1, "units")
                elif event.num == 5:
                    # Linux - wheel down
                    scrollable_widget.yview_scroll(1, "units")
            else:
                # Fallback: Trying to use self.canvas
                if hasattr(self, 'canvas'):
                    if event.delta:
                        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                    elif event.num == 4:
                        self.canvas.yview_scroll(-1, "units")
                    elif event.num == 5:
                        self.canvas.yview_scroll(1, "units")

        # We bind events to the entire main container and its descendants
        def _bind_recursive(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel)
            widget.bind("<Button-5>", _on_mousewheel)
            for child in widget.winfo_children():
                _bind_recursive(child)


        # ====== ACTOR SELECTION SECTION ======
        selection_frame = ttk.LabelFrame(self.scrollable_frame, text="🔍 Select Actor", padding=10)
        selection_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(selection_frame, text="Active Actor ID:").grid(row=0, column=0, sticky=tk.W)
        self.actor_id_display = ttk.Label(selection_frame, text="None", relief=tk.SUNKEN, foreground="red")
        self.actor_id_display.grid(row=0, column=1, sticky=tk.EW, padx=10)
        
        ttk.Label(selection_frame, text="Active Actor:").grid(row=1, column=0, sticky=tk.W)
        self.actor_name_display = ttk.Label(selection_frame, text="None", relief=tk.SUNKEN)
        self.actor_name_display.grid(row=1, column=1, sticky=tk.EW, padx=10)
        
        ttk.Button(selection_frame, text="🔄 Reload from Active", 
                   command=self.load_active_actor).grid(row=0, column=2, padx=5)
        ttk.Button(selection_frame, text="📖 History", 
                   command=self.show_history).grid(row=1, column=2, padx=5)
        
        selection_frame.columnconfigure(1, weight=1)
        
        # ====== OBSERVATION DATE SECTION ======
        date_frame = ttk.LabelFrame(self.scrollable_frame, text="📅 Observation Date", padding=10)
        date_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(date_frame, text="Observation Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W)
        self.obs_date_entry = ttk.Entry(date_frame, width=30)
        self.obs_date_entry.grid(row=0, column=1, sticky=tk.EW, padx=10)
        self.obs_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Button(date_frame, text="📆 Today", 
                   command=self.set_today_date).grid(row=0, column=2, padx=5)
        
        date_frame.columnconfigure(1, weight=1)
        
        # ====== RECORD STATUS SECTION ======
        status_frame = ttk.LabelFrame(self.scrollable_frame, text="📊 Record Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(status_frame, text="Record Status:").grid(row=0, column=0, sticky=tk.W)
        self.record_status_var = tk.StringVar(value='current')
        record_status_combo = ttk.Combobox(status_frame, textvariable=self.record_status_var,
                                           values=self.RECORD_STATUS_CHOICES, state='readonly', width=20)
        record_status_combo.grid(row=0, column=1, sticky=tk.EW, padx=10)
        
        ttk.Label(status_frame, text="(current = latest, superseded = outdated, rejected = invalid)").grid(row=0, column=2)
        
        status_frame.columnconfigure(1, weight=1)
        
        # ====== BIOGRAPHICAL INFORMATION ======
        bio_frame = ttk.LabelFrame(self.scrollable_frame, text="👤 Biographical Information", padding=10)
        bio_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = 0
        
        # Birth Place
        ttk.Label(bio_frame, text="Birth Place:").grid(row=row, column=0, sticky=tk.NW, pady=5)
        self.birth_place_entry = tk.Text(bio_frame, height=2, width=60)
        self.birth_place_entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        # Education
        ttk.Label(bio_frame, text="Education:").grid(row=row, column=0, sticky=tk.NW, pady=5)
        self.education_entry = tk.Text(bio_frame, height=3, width=60)
        self.education_entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        # Academic Titles
        ttk.Label(bio_frame, text="Academic Titles:").grid(row=row, column=0, sticky=tk.NW, pady=5)
        self.academic_titles_entry = ttk.Entry(bio_frame, width=60)
        self.academic_titles_entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        # Academic Works
        ttk.Label(bio_frame, text="Academic Works:").grid(row=row, column=0, sticky=tk.NW, pady=5)
        self.academic_works_entry = tk.Text(bio_frame, height=2, width=60)
        self.academic_works_entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        bio_frame.columnconfigure(1, weight=1)
        
        # ====== POLITICAL INFORMATION ======
        pol_frame = ttk.LabelFrame(self.scrollable_frame, text="🏛️ Political Information", padding=10)
        pol_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = 0
        
        ttk.Label(pol_frame, text="Political Party:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.political_party_entry = ttk.Entry(pol_frame, width=40)
        self.political_party_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        ttk.Label(pol_frame, text="Party Rank:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.party_rank_entry = ttk.Entry(pol_frame, width=40)
        self.party_rank_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        ttk.Label(pol_frame, text="Political Works:").grid(row=row, column=0, sticky=tk.NW, pady=5)
        self.political_works_entry = tk.Text(pol_frame, height=2, width=60)
        self.political_works_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        pol_frame.columnconfigure(1, weight=1)
        
        # ====== PROFESSIONAL INFORMATION ======
        prof_frame = ttk.LabelFrame(self.scrollable_frame, text="💼 Professional Information", padding=10)
        prof_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = 0
        
        ttk.Label(prof_frame, text="Position:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.prof_position_entry = ttk.Entry(prof_frame, width=40)
        self.prof_position_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        ttk.Label(prof_frame, text="Rank:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.prof_rank_entry = ttk.Entry(prof_frame, width=40)
        self.prof_rank_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        prof_frame.columnconfigure(1, weight=1)
        
        # ====== ELECTORAL INFORMATION ======
        elec_frame = ttk.LabelFrame(self.scrollable_frame, text="🗳️ Electoral Information", padding=10)
        elec_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = 0
        
        ttk.Label(elec_frame, text="Elections (comma-separated years):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.elections_entry = ttk.Entry(elec_frame, width=40)
        self.elections_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        ttk.Label(elec_frame, text="Status in Elected Bodies:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.elected_status_entry = ttk.Entry(elec_frame, width=40)
        self.elected_status_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        elec_frame.columnconfigure(1, weight=1)
        
        # ====== ORGANIZATIONAL & VISIBILITY ======
        org_frame = ttk.LabelFrame(self.scrollable_frame, text="🏢 Organizational & Visibility", padding=10)
        org_frame.pack(fill=tk.X, padx=10, pady=10)
        
        row = 0
        
        ttk.Label(org_frame, text="Affiliation (Organization):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.affiliation_entry = ttk.Entry(org_frame, width=40)
        self.affiliation_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        ttk.Label(org_frame, text="First Seen (YYYY-MM-DD):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.first_seen_entry = ttk.Entry(org_frame, width=40)
        self.first_seen_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        ttk.Label(org_frame, text="Last Seen (YYYY-MM-DD):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.last_seen_entry = ttk.Entry(org_frame, width=40)
        self.last_seen_entry.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        ttk.Label(org_frame, text="Current Status:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.status_var = tk.StringVar(value='active')
        status_combo = ttk.Combobox(org_frame, textvariable=self.status_var,
                                     values=self.STATUS_CHOICES, state='readonly', width=20)
        status_combo.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        row += 1
        
        org_frame.columnconfigure(1, weight=1)
        
        # ====== SOURCE & NOTES ======
        source_frame = ttk.LabelFrame(self.scrollable_frame, text="📝 Source & Notes", padding=10)
        source_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(source_frame, text="Source (Data Citation):").grid(row=0, column=0, sticky=tk.NW, pady=5)
        self.source_entry = tk.Text(source_frame, height=2, width=60)
        self.source_entry.grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        
        source_frame.columnconfigure(1, weight=1)
        
        # ====== ACTION BUTTONS ======
        action_frame = ttk.LabelFrame(self.scrollable_frame, text="⚙️ Actions", padding=10)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        button_container = ttk.Frame(action_frame)
        button_container.pack(fill=tk.X)
        
        ttk.Button(button_container, text="💾 Save Record", 
                   command=self.save_actor_record).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="🗑️ Delete Record", 
                   command=self.delete_current_record).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="🔄 Clear Form", 
                   command=self.clear_form).pack(side=tk.LEFT, padx=5)
        
        # ====== NAME ALIASES SECTION ======
        alias_frame = ttk.LabelFrame(self.scrollable_frame, text="📝 Name Aliases (Transliterations, Variants)", padding=10)
        alias_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Alias input
        alias_input_frame = ttk.LabelFrame(alias_frame, text="Add New Alias", padding=10)
        alias_input_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(alias_input_frame, text="Alias Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.alias_name_entry = ttk.Entry(alias_input_frame, width=30)
        self.alias_name_entry.grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(alias_input_frame, text="Type:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.alias_type_var = tk.StringVar(value='transliteration')
        alias_type_combo = ttk.Combobox(alias_input_frame, textvariable=self.alias_type_var,
                                         values=['transliteration', 'historical', 'nickname', 'variant'],
                                         state='readonly', width=25)
        alias_type_combo.grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(alias_input_frame, text="Language:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.alias_lang_var = tk.StringVar(value='en')
        alias_lang_combo = ttk.Combobox(alias_input_frame, textvariable=self.alias_lang_var,
                                         values=['en', 'ru', 'zh_prc', 'zh_hk', 'pinyin'],
                                         state='readonly', width=25)
        alias_lang_combo.grid(row=2, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Label(alias_input_frame, text="Confidence:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.alias_confidence_var = tk.StringVar(value='0.95')
        confidence_spin = ttk.Spinbox(alias_input_frame, from_=0.0, to=1.0, increment=0.05,
                                      textvariable=self.alias_confidence_var, width=25)
        confidence_spin.grid(row=3, column=1, sticky=tk.EW, padx=10, pady=5)
        
        ttk.Button(alias_input_frame, text="➕ Add Alias", 
                   command=self.add_alias).grid(row=4, column=0, columnspan=2, pady=10)
        
        alias_input_frame.columnconfigure(1, weight=1)
        
        # Aliases list
        alias_list_frame = ttk.LabelFrame(alias_frame, text="Current Aliases", padding=10)
        alias_list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create Treeview for aliases
        self.aliases_tree = ttk.Treeview(alias_list_frame, columns=('Name', 'Type', 'Language', 'Confidence', 'Source'),
                                          height=6, show='tree headings')
        
        self.aliases_tree.column('#0', width=0, stretch=tk.NO)
        self.aliases_tree.column('Name', anchor=tk.W, width=150)
        self.aliases_tree.column('Type', anchor=tk.W, width=100)
        self.aliases_tree.column('Language', anchor=tk.W, width=80)
        self.aliases_tree.column('Confidence', anchor=tk.CENTER, width=80)
        self.aliases_tree.column('Source', anchor=tk.W, width=150)
        
        self.aliases_tree.heading('Name', text='Alias Name')
        self.aliases_tree.heading('Type', text='Type')
        self.aliases_tree.heading('Language', text='Language')
        self.aliases_tree.heading('Confidence', text='Confidence')
        self.aliases_tree.heading('Source', text='Source')
        
        # Scrollbar for tree
        alias_scroll = ttk.Scrollbar(alias_list_frame, orient='vertical', command=self.aliases_tree.yview)
        self.aliases_tree.configure(yscroll=alias_scroll.set)
        
        self.aliases_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alias_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Delete button for aliases
        ttk.Button(alias_frame, text="🗑️ Delete Selected Alias", 
                   command=self.delete_selected_alias).pack(fill=tk.X, pady=10)
   
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind the configure event to update the canvas width
        def on_configure(event):
            self.canvas.itemconfig(1, width=event.width)

        self.canvas.bind("<Configure>", on_configure)
                
        # IMPORTANT: Bind scroll events AFTER creating all widgets
        _bind_recursive(main_container)
    
    def load_active_actor(self):
        """Load actor data from active_actor_id in session"""
        if not self.session.active_actor_id:
            messagebox.showwarning("No Active Actor", "Please create an actor first in '👤 Create Actor ID' tab")
            return
        
        try:
            # Get actor ID info
            actor_info = self.actor_manager.get_actor_id_info(self.session.active_actor_id)
            
            if not actor_info:
                messagebox.showerror("Error", "Actor ID not found")
                return
            
            # Display actor info
            self.actor_id_display.config(text=self.session.active_actor_id[:16] + "...", foreground="green")
            
            # Get actor's name (from appropriate field)
            name = (actor_info.get('name_ru') or actor_info.get('name_zh_prc') or 
                   actor_info.get('name_zh_hk') or actor_info.get('name_en') or "Unknown")
            self.actor_name_display.config(text=name)
            
            # Load current biographical record
            current_record = self.actor_manager.get_current_actor_status(self.session.active_actor_id)
            
            if current_record:
                self.populate_form(current_record)
                self.current_record_id = current_record['record_id']
            else:
                self.clear_form()
                self.current_record_id = None
            
            # Load aliases
            self.load_aliases()
            
            self.session.mark_unsaved('Actors')
            logger.info(f"Loaded actor: {self.session.active_actor_id}")
            
        except Exception as e:
            logger.error(f"Error loading actor: {e}")
            messagebox.showerror("Error", str(e))
    
    def populate_form(self, record: Dict):
        """Populate form fields with record data"""
        self.birth_place_entry.delete("1.0", tk.END)
        self.birth_place_entry.insert("1.0", record.get('birth_place', '') or '')
        
        self.education_entry.delete("1.0", tk.END)
        self.education_entry.insert("1.0", record.get('education', '') or '')
        
        self.academic_titles_entry.delete(0, tk.END)
        self.academic_titles_entry.insert(0, record.get('academic_titles', '') or '')
        
        self.academic_works_entry.delete("1.0", tk.END)
        self.academic_works_entry.insert("1.0", record.get('academic_works', '') or '')
        
        self.political_party_entry.delete(0, tk.END)
        self.political_party_entry.insert(0, record.get('political_party', '') or '')
        
        self.party_rank_entry.delete(0, tk.END)
        self.party_rank_entry.insert(0, record.get('party_rank', '') or '')
        
        self.political_works_entry.delete("1.0", tk.END)
        self.political_works_entry.insert("1.0", record.get('political_works', '') or '')
        
        self.prof_position_entry.delete(0, tk.END)
        self.prof_position_entry.insert(0, record.get('prof_position', '') or '')
        
        self.prof_rank_entry.delete(0, tk.END)
        self.prof_rank_entry.insert(0, record.get('prof_rank', '') or '')
        
        self.elections_entry.delete(0, tk.END)
        self.elections_entry.insert(0, record.get('participation_in_elections', '') or '')
        
        self.elected_status_entry.delete(0, tk.END)
        self.elected_status_entry.insert(0, record.get('status_in_elected_bodies', '') or '')
        
        self.affiliation_entry.delete(0, tk.END)
        self.affiliation_entry.insert(0, record.get('affiliation', '') or '')
        
        self.first_seen_entry.delete(0, tk.END)
        if record.get('first_seen'):
            self.first_seen_entry.insert(0, str(record['first_seen']))
        
        self.last_seen_entry.delete(0, tk.END)
        if record.get('last_seen'):
            self.last_seen_entry.insert(0, str(record['last_seen']))
        
        self.status_var.set(record.get('status', 'active'))
        
        self.record_status_var.set(record.get('record_status', 'current'))
        
        self.obs_date_entry.delete(0, tk.END)
        self.obs_date_entry.insert(0, str(record.get('observation_date', datetime.now().strftime('%Y-%m-%d'))))
        
        self.source_entry.delete("1.0", tk.END)
        self.source_entry.insert("1.0", record.get('source', '') or '')
        
        logger.info(f"Form populated with record: {record.get('record_id')}")
    
    def clear_form(self):
        """Clear all form fields"""
        for widget in [self.birth_place_entry, self.education_entry, self.academic_works_entry,
                       self.political_works_entry, self.source_entry]:
            if isinstance(widget, tk.Text):
                widget.delete("1.0", tk.END)
        
        for widget in [self.academic_titles_entry, self.political_party_entry, self.party_rank_entry,
                       self.prof_position_entry, self.prof_rank_entry, self.elections_entry,
                       self.elected_status_entry, self.affiliation_entry, self.first_seen_entry,
                       self.last_seen_entry]:
            if isinstance(widget, ttk.Entry):
                widget.delete(0, tk.END)
        
        self.status_var.set('active')
        self.record_status_var.set('current')
        self.obs_date_entry.delete(0, tk.END)
        self.obs_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        self.current_record_id = None
        logger.info("Form cleared")
    
    def set_today_date(self):
        """Set observation date to today"""
        self.obs_date_entry.delete(0, tk.END)
        self.obs_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
    
    def save_actor_record(self):
        """Save biographical record to database"""
        if not self.session.active_actor_id:
            messagebox.showwarning("No Active Actor", "Please load an actor first")
            return
        
        try:
            # Collect form data
            obs_date = self.obs_date_entry.get()
            
            # Validate date format
            try:
                datetime.strptime(obs_date, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("Invalid Date", "Please use YYYY-MM-DD format")
                return
            
            biographical_data = {
                'observation_date': obs_date,
                'record_status': self.record_status_var.get(),
                'birth_place': self.birth_place_entry.get("1.0", tk.END).strip() or None,
                'education': self.education_entry.get("1.0", tk.END).strip() or None,
                'academic_titles': self.academic_titles_entry.get() or None,
                'academic_works': self.academic_works_entry.get("1.0", tk.END).strip() or None,
                'political_party': self.political_party_entry.get() or None,
                'party_rank': self.party_rank_entry.get() or None,
                'political_works': self.political_works_entry.get("1.0", tk.END).strip() or None,
                'prof_position': self.prof_position_entry.get() or None,
                'prof_rank': self.prof_rank_entry.get() or None,
                'participation_in_elections': self.elections_entry.get() or None,
                'status_in_elected_bodies': self.elected_status_entry.get() or None,
                'affiliation': self.affiliation_entry.get() or None,
                'first_seen': self.first_seen_entry.get() or None,
                'last_seen': self.last_seen_entry.get() or None,
                'status': self.status_var.get(),
                'source': self.source_entry.get("1.0", tk.END).strip() or None
            }
            
            # Save to database
            record_id = self.actor_manager.add_actor_record(self.session.active_actor_id, **biographical_data)
            
            messagebox.showinfo("Success", f"✓ Biographical record saved (ID: {record_id})")
            self.current_record_id = record_id
            self.session.clear_unsaved('Actors')
            
            logger.info(f"Saved actor record: {record_id}")
            
        except Exception as e:
            logger.error(f"Error saving record: {e}")
            messagebox.showerror("Error", str(e))
    
    def delete_current_record(self):
        """Delete current biographical record"""
        if not self.current_record_id:
            messagebox.showinfo("No Record", "No record selected to delete")
            return
        
        if not messagebox.askyesno("Confirm Delete", "Delete this biographical record?"):
            return
        
        try:
            query = 'DELETE FROM Actors WHERE record_id = ?'
            self.db.execute_insert(query, (self.current_record_id,))
            
            messagebox.showinfo("Success", "✓ Record deleted")
            self.clear_form()
            self.current_record_id = None
            
            logger.info(f"Deleted record: {self.current_record_id}")
            
        except Exception as e:
            logger.error(f"Error deleting record: {e}")
            messagebox.showerror("Error", str(e))
    
    def load_aliases(self):
        """Load and display actor aliases"""
        if not self.session.active_actor_id:
            return
        
        # Clear tree
        for item in self.aliases_tree.get_children():
            self.aliases_tree.delete(item)
        
        try:
            aliases = self.actor_manager.get_actor_aliases(self.session.active_actor_id)
            self.current_aliases = aliases
            
            for alias in aliases:
                self.aliases_tree.insert('', 'end',
                                         values=(alias['alias_name'],
                                                alias['alias_type'],
                                                alias['language_code'] or '',
                                                f"{alias['confidence']:.2f}",
                                                alias['source'] or ''))
            
            logger.info(f"Loaded {len(aliases)} aliases")
            
        except Exception as e:
            logger.error(f"Error loading aliases: {e}")
    
    def add_alias(self):
        """Add new name alias"""
        if not self.session.active_actor_id:
            messagebox.showwarning("No Active Actor", "Please load an actor first")
            return
        
        alias_name = self.alias_name_entry.get().strip()
        if not alias_name:
            messagebox.showwarning("Missing Data", "Alias name is required")
            return
        
        try:
            confidence = float(self.alias_confidence_var.get())
            
            self.actor_manager.add_actor_alias(
                actor_id=self.session.active_actor_id,
                alias_name=alias_name,
                alias_type=self.alias_type_var.get(),
                language_code=self.alias_lang_var.get(),
                confidence=confidence,
                source=None
            )
            
            self.alias_name_entry.delete(0, tk.END)
            self.load_aliases()
            messagebox.showinfo("Success", "✓ Alias added")
            
            logger.info(f"Added alias: {alias_name}")
            
        except ValueError:
            messagebox.showerror("Invalid Confidence", "Confidence must be a number 0-1")
        except Exception as e:
            logger.error(f"Error adding alias: {e}")
            messagebox.showerror("Error", str(e))
    
    def delete_selected_alias(self):
        """Delete selected alias"""
        selection = self.aliases_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select an alias to delete")
            return
        
        if not messagebox.askyesno("Confirm", "Delete this alias?"):
            return
        
        try:
            # Get selected alias
            item_index = self.aliases_tree.index(selection[0])
            alias_id = self.current_aliases[item_index]['alias_id']
            
            self.actor_manager.delete_actor_alias(alias_id)
            self.load_aliases()
            messagebox.showinfo("Success", "✓ Alias deleted")
            
            logger.info(f"Deleted alias: {alias_id}")
            
        except Exception as e:
            logger.error(f"Error deleting alias: {e}")
            messagebox.showerror("Error", str(e))
    
    def show_history(self):
        """Show full actor history"""
        if not self.session.active_actor_id:
            messagebox.showwarning("No Active Actor", "Please load an actor first")
            return
        
        try:
            history = self.actor_manager.get_actor_records_history(self.session.active_actor_id)
            
            if not history:
                messagebox.showinfo("History", "No biographical records found")
                return
            
            # Create history window
            history_window = tk.Toplevel(self.parent)
            history_window.title(f"Actor History - {self.session.active_actor_id[:16]}...")
            history_window.geometry("900x500")
            
            # Create tree
            tree = ttk.Treeview(history_window,
                               columns=('Record ID', 'Date', 'Status', 'Position', 'Status Change', 'Updated'),
                               height=15, show='tree headings')
            
            tree.column('#0', width=0, stretch=tk.NO)
            tree.column('Record ID', anchor=tk.W, width=80)
            tree.column('Date', anchor=tk.W, width=100)
            tree.column('Status', anchor=tk.W, width=100)
            tree.column('Position', anchor=tk.W, width=200)
            tree.column('Status Change', anchor=tk.W, width=100)
            tree.column('Updated', anchor=tk.W, width=120)
            
            tree.heading('Record ID', text='Record ID')
            tree.heading('Date', text='Observation Date')
            tree.heading('Status', text='Record Status')
            tree.heading('Position', text='Position')
            tree.heading('Status Change', text='Status')
            tree.heading('Updated', text='Updated')
            
            for record in history:
                tree.insert('', 'end',
                           values=(record['record_id'],
                                  record['observation_date'],
                                  record['record_status'],
                                  record.get('prof_position', '')[:50],
                                  record.get('status', ''),
                                  record['updated_at'][:19]))
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(history_window, orient='vertical', command=tree.yview)
            tree.configure(yscroll=scrollbar.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
        except Exception as e:
            logger.error(f"Error showing history: {e}")
            messagebox.showerror("Error", str(e))


class EdgesTab:
    """Complete implementation of 'Edges/Relationships' tab"""

    RELATIONSHIP_TYPES = [
        'kinship', 'trust', 'community', 'informal',
        'official', 'mentoring', 'communicative', 'transactional',
        'educational', 'conflict'
    ]

    def __init__(self, parent_frame, edge_manager, actor_manager, name_lookup,
                 session, db):
        """Initialize Edges tab"""
        self.edge_manager = edge_manager
        self.actor_manager = actor_manager
        self.name_lookup = name_lookup
        self.session = session
        self.db = db
        self.parent = parent_frame

        # Track current edge being created
        self.current_source_actor_id = None
        self.current_source_actor_name = None
        self.current_target_actor_id = None
        self.current_target_actor_name = None

        self.create_ui()
        logger.info("Edges tab initialized")

    def create_ui(self):
        """Create complete UI for edges tab"""

        # Main container
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ====== FIELD 1: SOURCE ACTOR SECTION ======
        source_frame = ttk.LabelFrame(main_frame, text="📤 Source Actor", padding=10)
        source_frame.pack(fill=tk.X, pady=10)

        ttk.Label(source_frame, text="Source Actor ID:").grid(row=0, column=0, sticky=tk.W)
        self.source_id_display = ttk.Label(source_frame, text="None", relief=tk.SUNKEN,
                                           foreground="red", background="#fff0f0")
        self.source_id_display.grid(row=0, column=1, sticky=tk.EW, padx=10, ipady=5)

        ttk.Label(source_frame, text="Source Actor Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.source_name_display = ttk.Label(source_frame, text="None", relief=tk.SUNKEN)
        self.source_name_display.grid(row=1, column=1, sticky=tk.EW, padx=10, ipady=5)

        button_frame1 = ttk.Frame(source_frame)
        button_frame1.grid(row=0, column=2, rowspan=2, sticky=tk.EW, padx=10)
        ttk.Button(button_frame1, text="🔄 Change Active",
                  command=self.change_active_actor).pack(side=tk.TOP, pady=2)
        ttk.Button(button_frame1, text="🔍 Select Source",
                  command=self.select_source_actor).pack(side=tk.TOP, pady=2)

        source_frame.columnconfigure(1, weight=1)

        # Auto-populate from session if available
        self.update_source_from_session()

        # ====== FIELD 2: TARGET ACTOR SECTION ======
        target_frame = ttk.LabelFrame(main_frame, text="📥 Target Actor", padding=10)
        target_frame.pack(fill=tk.X, pady=10)

        ttk.Label(target_frame, text="Target Actor ID:").grid(row=0, column=0, sticky=tk.W)
        self.target_id_display = ttk.Label(target_frame, text="None", relief=tk.SUNKEN,
                                           foreground="red", background="#fff0f0")
        self.target_id_display.grid(row=0, column=1, sticky=tk.EW, padx=10, ipady=5)

        ttk.Label(target_frame, text="Target Actor Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.target_name_display = ttk.Label(target_frame, text="None", relief=tk.SUNKEN)
        self.target_name_display.grid(row=1, column=1, sticky=tk.EW, padx=10, ipady=5)

        button_frame2 = ttk.Frame(target_frame)
        button_frame2.grid(row=0, column=2, rowspan=2, sticky=tk.EW, padx=10)
        ttk.Button(button_frame2, text="🔍 Select Target",
                  command=self.select_target_actor).pack(side=tk.TOP, pady=2)

        target_frame.columnconfigure(1, weight=1)

        # ====== FIELD 3: RELATIONSHIP TYPE SECTION ======
        type_frame = ttk.LabelFrame(main_frame, text="🔗 Relationship Type", padding=10)
        type_frame.pack(fill=tk.X, pady=10)

        ttk.Label(type_frame, text="Select relationship type:").grid(row=0, column=0, sticky=tk.W)
        self.relationship_type_var = tk.StringVar()
        type_combo = ttk.Combobox(type_frame, textvariable=self.relationship_type_var,
                                  values=self.RELATIONSHIP_TYPES, state='readonly', width=30)
        type_combo.grid(row=0, column=1, sticky=tk.EW, padx=10, ipady=5)

        # Type descriptions
        descriptions = {
            'kinship': 'Family relations',
            'trust': 'Trust-based relationships',
            'community': 'Shared community/group membership',
            'informal': 'Informal connections',
            'official': 'Official hierarchical relationships',
            'mentoring': 'Mentor-mentee relationships',
            'communicative': 'Direct communication channels',
            'transactional': 'Business/financial transactions',
            'educational': 'Education/training relationships',
            'conflict': 'Adversarial relationships'
        }

        self.type_description_label = ttk.Label(type_frame, text="", foreground="gray")
        self.type_description_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

        def on_type_change(event=None):
            selected = self.relationship_type_var.get()
            desc = descriptions.get(selected, "")
            self.type_description_label.config(text=f"ℹ️ {desc}")

        type_combo.bind('<<ComboboxSelected>>', on_type_change)

        type_frame.columnconfigure(1, weight=1)

        # ====== FIELD 4: ACTION BUTTONS ======
        action_frame = ttk.LabelFrame(main_frame, text="⚙️ Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=10)

        button_container = ttk.Frame(action_frame)
        button_container.pack(fill=tk.X)

        ttk.Button(button_container, text="💾 Save Edge",
                  command=self.save_edge).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="🗑️ Reset",
                  command=self.reset_form).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="📊 View Edges",
                  command=self.view_edges).pack(side=tk.LEFT, padx=5)

        # ====== EDGES LIST (Paginated Table) ======
        list_frame = ttk.LabelFrame(main_frame, text="📋 Recent Edges", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.edges_tree = ttk.Treeview(list_frame,
                                       columns=('ID', 'Source', 'Target', 'Type', 'Date'),
                                       height=10, show='tree headings')

        self.edges_tree.column('#0', width=0, stretch=tk.NO)
        self.edges_tree.column('ID', anchor=tk.CENTER, width=60)
        self.edges_tree.column('Source', anchor=tk.W, width=200)
        self.edges_tree.column('Target', anchor=tk.W, width=200)
        self.edges_tree.column('Type', anchor=tk.W, width=120)
        self.edges_tree.column('Date', anchor=tk.CENTER, width=140)

        self.edges_tree.heading('ID', text='Edge ID')
        self.edges_tree.heading('Source', text='Source Actor')
        self.edges_tree.heading('Target', text='Target Actor')
        self.edges_tree.heading('Type', text='Relationship Type')
        self.edges_tree.heading('Date', text='Created')

        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.edges_tree.yview)
        self.edges_tree.configure(yscroll=scrollbar.set)

        self.edges_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind right-click delete
        self.edges_tree.bind('<Button-3>', self.on_edge_right_click)

        # Pagination controls
        pagination_frame = ttk.Frame(main_frame)
        pagination_frame.pack(fill=tk.X, pady=10)

        ttk.Button(pagination_frame, text="🔄 Refresh List",
                  command=self.refresh_edges_list).pack(side=tk.LEFT, padx=5)

        self.edges_count_label = ttk.Label(pagination_frame, text="Total edges: 0")
        self.edges_count_label.pack(side=tk.RIGHT, padx=5)

        # Initial load
        self.refresh_edges_list()

    # ====== HELPER METHODS ======

    def update_source_from_session(self):
        """Auto-populate source from active actor in session"""
        if self.session.active_actor_id:
            self.current_source_actor_id = self.session.active_actor_id
            actor_info = self.actor_manager.get_actor_id_info(self.session.active_actor_id)
            if actor_info:
                name_fields = ['name_zh_prc', 'name_ru', 'name_zh_hk', 'name_en']
                name = next((actor_info.get(f) for f in name_fields if actor_info.get(f)), "Unknown")
                self.current_source_actor_name = name

            self.source_id_display.config(text=self.current_source_actor_id[:12] + "...",
                                         foreground="green")
            self.source_name_display.config(text=self.current_source_actor_name)

    def change_active_actor(self):
        """Open modal to change active actor"""
        modal = SelectActorModal(self.parent, self.name_lookup, self.actor_manager, self.db,
                                "Change Active Actor")
        self.parent.wait_window(modal)

        if modal.selected_actor_id:
            self.session.active_actor_id = modal.selected_actor_id
            self.update_source_from_session()

    def select_source_actor(self):
        """Open modal to select source actor"""
        modal = SelectActorModal(self.parent, self.name_lookup, self.actor_manager, self.db,
                                "Select Source Actor")
        self.parent.wait_window(modal)

        if modal.selected_actor_id:
            self.current_source_actor_id = modal.selected_actor_id
            self.current_source_actor_name = modal.selected_actor_name

            self.source_id_display.config(text=self.current_source_actor_id[:12] + "...",
                                         foreground="green")
            self.source_name_display.config(text=self.current_source_actor_name)

    def select_target_actor(self):
        """Open modal to select target actor"""
        modal = SelectActorModal(self.parent, self.name_lookup, self.actor_manager, self.db,
                                "Select Target Actor")
        self.parent.wait_window(modal)

        if modal.selected_actor_id:
            self.current_target_actor_id = modal.selected_actor_id
            self.current_target_actor_name = modal.selected_actor_name

            self.target_id_display.config(text=self.current_target_actor_id[:12] + "...",
                                         foreground="green")
            self.target_name_display.config(text=self.current_target_actor_name)

    def save_edge(self):
        """Save edge with validation"""
        try:
            # Get values
            source_id = self.current_source_actor_id
            target_id = self.current_target_actor_id
            rel_type = self.relationship_type_var.get()

            # Validate
            is_valid, error_msg = self.edge_manager.validate_edge(source_id, target_id, rel_type)

            if not is_valid:
                messagebox.showerror("Validation Error", error_msg)
                return

            # Create edge
            edge_id = self.edge_manager.create_edge(source_id, target_id, rel_type)

            messagebox.showinfo("Success",
                              f"✓ Edge created successfully!\nEdge ID: {edge_id}\n\nSource: {self.current_source_actor_name}\nTarget: {self.current_target_actor_name}\nType: {rel_type}")

            logger.info(f"Edge created: {edge_id}")

            # Refresh and reset
            self.refresh_edges_list()
            self.reset_form()

        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
        except Exception as e:
            logger.error(f"Error saving edge: {e}")
            messagebox.showerror("Error", f"Error saving edge: {str(e)}")

    def reset_form(self):
        """Clear form fields"""
        self.current_target_actor_id = None
        self.current_target_actor_name = None

        self.target_id_display.config(text="None", foreground="red")
        self.target_name_display.config(text="None")
        self.relationship_type_var.set("")
        self.type_description_label.config(text="")

    def refresh_edges_list(self):
        """Refresh the edges list display"""
        for item in self.edges_tree.get_children():
            self.edges_tree.delete(item)

        try:
            edges = self.edge_manager.get_all_edges(limit=100)

            for i, edge in enumerate(edges):
                edge_id = edge['edge_id']
                source_id = edge['source_actor_id']
                target_id = edge['target_actor_id']
                rel_type = edge['relationship_type']
                timestamp = edge['created_timestamp'][:10]  # Date only

                # Get actor names
                source_info = self.actor_manager.get_actor_id_info(source_id)
                target_info = self.actor_manager.get_actor_id_info(target_id)

                name_fields = ['name_zh_prc', 'name_ru', 'name_zh_hk', 'name_en']
                source_name = next((source_info.get(f) for f in name_fields if source_info and source_info.get(f)), source_id[:12])
                target_name = next((target_info.get(f) for f in name_fields if target_info and target_info.get(f)), target_id[:12])

                self.edges_tree.insert('', 'end', iid=f"edge_{edge_id}",
                                       values=(edge_id, source_name, target_name, rel_type, timestamp))

            count = self.edge_manager.get_edge_count()
            self.edges_count_label.config(text=f"Total edges: {count}")

        except Exception as e:
            logger.error(f"Error refreshing edges: {e}")
            messagebox.showerror("Error", f"Error loading edges: {str(e)}")

    def on_edge_right_click(self, event):
        """Handle right-click on edge (delete)"""
        item = self.edges_tree.selection()[0] if self.edges_tree.selection() else None
        if not item:
            return

        if messagebox.askyesno("Delete Edge", "Delete this edge?"):
            edge_id = int(item.split('_')[1])
            try:
                self.edge_manager.delete_edge(edge_id)
                self.refresh_edges_list()
                messagebox.showinfo("Deleted", f"Edge {edge_id} deleted")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def view_edges(self):
        """Show detailed edges viewer window"""
        # This would open a new window with filtering and sorting options
        messagebox.showinfo("Edges Viewer", "Advanced viewer - use right-click on edges to delete")


class EventsTab:
    """Complete implementation of Events tab"""
    
    DOMAINS = ['politics', 'economics', 'society', 'security', 'technology',
               'healthcare', 'culture', 'sports', 'nature', 'unknown']
    
    TYPES = ['trigger', 'process', 'institutional', 'normative',
             'symbol_narrative', 'forecast', 'unknown']
    
    SCALES = ['local', 'national', 'regional', 'global', 'unknown']
    
    PRIORITIES = ['strategic', 'tactical', 'noise', 'unknown']
    
    PRECISIONS = ['day', 'month', 'year', 'unknown']
    
    def __init__(self, parent_frame, event_manager, history_manager, db):
        """Initialize Events tab"""
        self.event_manager = event_manager
        self.history_manager = history_manager
        self.db = db
        self.parent = parent_frame
        
        self.current_event_id = None
        self.form_vars = {}
        
        self.create_ui()
        logger.info("Events tab initialized")
    
    def create_ui(self):
        """Create complete UI for events tab"""
        
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=9, pady=9)
        
        # ===== TOP BUTTON =====
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="🔍 Select Event", 
                  command=self.open_search_dialog).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="🗑️ Clear Form", 
                  command=self.clear_form).pack(side=tk.LEFT, padx=5)
        
        # ===== FORM FIELDS =====
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for form
        canvas = tk.Canvas(form_frame, bg='white')
        scrollbar = ttk.Scrollbar(form_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)


        # Update canvas width when window size changes
        def on_frame_configure(event):
            canvas.itemconfig(1, width=event.width)
        form_frame.bind("<Configure>", on_frame_configure)

        
        # Row counter for form layout
        row = 0
        
        # ===== EVENT TITLES =====
        ttk.Label(scrollable_frame, text="Event Title (RU):", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['event_title_ru'] = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.form_vars['event_title_ru'], width=80).grid(row=row, column=1, sticky=tk.EW, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Event Title (ZH):", font=('', 14)).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['event_title_zh'] = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.form_vars['event_title_zh'], width=80).grid(row=row, column=1, sticky=tk.EW, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Event Title (EN):", font=('', 14)).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['event_title_en'] = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.form_vars['event_title_en'], width=80).grid(row=row, column=1, sticky=tk.EW, padx=10)
        row += 1
        
        # ===== EVENT DATE =====
        ttk.Label(scrollable_frame, text="Event Date (YYYY-MM-DD):", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['event_date'] = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.form_vars['event_date'], width=80).grid(row=row, column=1, sticky=tk.EW, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Date Precision:", font=('', 14)).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['event_date_precision'] = tk.StringVar(value='day')
        ttk.Combobox(scrollable_frame, textvariable=self.form_vars['event_date_precision'], 
                    values=self.PRECISIONS, state='readonly', width=20).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        # ===== DESCRIPTIONS =====
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Description (RU):", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.NW, padx=10, pady=5)
        self.form_vars['description_ru'] = tk.StringVar()
        text_ru = scrolledtext.ScrolledText(scrollable_frame, height=4, width=80, wrap=tk.WORD)
        text_ru.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        self.form_vars['description_ru_widget'] = text_ru
        row += 1
        
        ttk.Label(scrollable_frame, text="Description (ZH):", font=('', 14)).grid(row=row, column=0, sticky=tk.NW, padx=10, pady=5)
        self.form_vars['description_zh'] = tk.StringVar()
        text_zh = scrolledtext.ScrolledText(scrollable_frame, height=4, width=80, wrap=tk.WORD)
        text_zh.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        self.form_vars['description_zh_widget'] = text_zh
        row += 1
        
        ttk.Label(scrollable_frame, text="Description (EN):", font=('', 14)).grid(row=row, column=0, sticky=tk.NW, padx=10, pady=5)
        self.form_vars['description_en'] = tk.StringVar()
        text_en = scrolledtext.ScrolledText(scrollable_frame, height=4, width=80, wrap=tk.WORD)
        text_en.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        self.form_vars['description_en_widget'] = text_en
        row += 1
        
        # ===== CLASSIFICATION FIELDS =====
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Event Domain:", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['event_domain'] = tk.StringVar(value='unknown')
        ttk.Combobox(scrollable_frame, textvariable=self.form_vars['event_domain'], 
                    values=self.DOMAINS, state='readonly', width=20).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Event Type:", font=('', 14)).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['event_type'] = tk.StringVar(value='unknown')
        ttk.Combobox(scrollable_frame, textvariable=self.form_vars['event_type'], 
                    values=self.TYPES, state='readonly', width=20).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Event Scale:", font=('', 14)).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['event_scale'] = tk.StringVar(value='unknown')
        ttk.Combobox(scrollable_frame, textvariable=self.form_vars['event_scale'], 
                    values=self.SCALES, state='readonly', width=20).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Event Priority:", font=('', 14)).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['event_priority'] = tk.StringVar(value='unknown')
        ttk.Combobox(scrollable_frame, textvariable=self.form_vars['event_priority'], 
                    values=self.PRIORITIES, state='readonly', width=20).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        # ===== OTHER FIELDS =====
        ttk.Label(scrollable_frame, text="Participants Count:", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
        self.form_vars['participants_count'] = tk.StringVar()
        ttk.Entry(scrollable_frame, textvariable=self.form_vars['participants_count'], width=20).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Sources:", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.NW, padx=10, pady=5)
        self.form_vars['sources'] = tk.StringVar()
        text_sources = scrolledtext.ScrolledText(scrollable_frame, height=4, width=80, wrap=tk.WORD)
        text_sources.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        self.form_vars['sources_widget'] = text_sources
        row += 1
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ===== SAVE BUTTON =====
        save_frame = ttk.Frame(main_frame)
        save_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(save_frame, text="💾 Save Event", 
                  command=self.save_event).pack(side=tk.LEFT, padx=5)
        
        # Configure column weights for resizing
        scrollable_frame.columnconfigure(1, weight=1)
    
    def open_search_dialog(self):
        """Open event search modal"""
        modal = SearchEventsModal(self.parent, self.event_manager, self.history_manager)
        self.parent.wait_window(modal)
        
        if modal.selected_event:
            self.load_event_to_form(modal.selected_event)
            self.current_event_id = modal.selected_event['event_id']
    
    def load_event_to_form(self, event_data: Dict):
        """Load event data into form fields"""
        self.form_vars['event_title_ru'].set(event_data.get('event_title_ru', ''))
        self.form_vars['event_title_zh'].set(event_data.get('event_title_zh', ''))
        self.form_vars['event_title_en'].set(event_data.get('event_title_en', ''))
        
        self.form_vars['event_date'].set(event_data.get('event_date', ''))
        self.form_vars['event_date_precision'].set(event_data.get('event_date_precision', 'day'))
        
        self.form_vars['description_ru_widget'].delete('1.0', tk.END)
        self.form_vars['description_ru_widget'].insert('1.0', event_data.get('description_ru', ''))
        
        self.form_vars['description_zh_widget'].delete('1.0', tk.END)
        self.form_vars['description_zh_widget'].insert('1.0', event_data.get('description_zh', ''))
        
        self.form_vars['description_en_widget'].delete('1.0', tk.END)
        self.form_vars['description_en_widget'].insert('1.0', event_data.get('description_en', ''))
        
        self.form_vars['event_domain'].set(event_data.get('event_domain', 'unknown'))
        self.form_vars['event_type'].set(event_data.get('event_type', 'unknown'))
        self.form_vars['event_scale'].set(event_data.get('event_scale', 'unknown'))
        self.form_vars['event_priority'].set(event_data.get('event_priority', 'unknown'))
        
        self.form_vars['participants_count'].set(str(event_data.get('participants_count', '')))
        
        self.form_vars['sources_widget'].delete('1.0', tk.END)
        self.form_vars['sources_widget'].insert('1.0', event_data.get('sources', ''))
    
    def get_form_data(self) -> Dict:
        """Collect form data into dictionary"""
        return {
            'event_title_ru': self.form_vars['event_title_ru'].get(),
            'event_title_zh': self.form_vars['event_title_zh'].get(),
            'event_title_en': self.form_vars['event_title_en'].get(),
            'event_date': self.form_vars['event_date'].get(),
            'event_date_precision': self.form_vars['event_date_precision'].get(),
            'description_ru': self.form_vars['description_ru_widget'].get('1.0', tk.END).strip(),
            'description_zh': self.form_vars['description_zh_widget'].get('1.0', tk.END).strip(),
            'description_en': self.form_vars['description_en_widget'].get('1.0', tk.END).strip(),
            'event_domain': self.form_vars['event_domain'].get(),
            'event_type': self.form_vars['event_type'].get(),
            'event_scale': self.form_vars['event_scale'].get(),
            'event_priority': self.form_vars['event_priority'].get(),
            'participants_count': self.form_vars['participants_count'].get(),
            'sources': self.form_vars['sources_widget'].get('1.0', tk.END).strip()
        }
    
    def save_event(self):
        """Save event to database"""
        try:
            event_data = self.get_form_data()
            
            if self.current_event_id:
                # Update existing event
                self.event_manager.update_event(self.current_event_id, event_data)
                messagebox.showinfo("Success", f"Event {self.current_event_id} updated successfully")
                logger.info(f"Updated event {self.current_event_id}")
            else:
                # Create new event
                event_id = self.event_manager.create_event(event_data)
                self.history_manager.add_search(event_data)
                messagebox.showinfo("Success", f"Event created successfully (ID: {event_id})")
                logger.info(f"Created event {event_id}")
                self.current_event_id = event_id
            
            self.clear_form()
        
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
        except Exception as e:
            logger.error(f"Error saving event: {e}")
            messagebox.showerror("Error", f"Error saving event: {str(e)}")
    
    def clear_form(self):
        """Clear all form fields"""
        for var_name, var in self.form_vars.items():
            if isinstance(var, tk.StringVar):
                var.set('')
        
        # Clear text widgets
        if 'description_ru_widget' in self.form_vars:
            self.form_vars['description_ru_widget'].delete('1.0', tk.END)
        if 'description_zh_widget' in self.form_vars:
            self.form_vars['description_zh_widget'].delete('1.0', tk.END)
        if 'description_en_widget' in self.form_vars:
            self.form_vars['description_en_widget'].delete('1.0', tk.END)
        if 'sources_widget' in self.form_vars:
            self.form_vars['sources_widget'].delete('1.0', tk.END)
        
        # Reset dropdown defaults
        self.form_vars['event_date_precision'].set('day')
        self.form_vars['event_domain'].set('unknown')
        self.form_vars['event_type'].set('unknown')
        self.form_vars['event_scale'].set('unknown')
        self.form_vars['event_priority'].set('unknown')
        
        self.current_event_id = None


class ParticipationTab:
    """Complete implementation of Participation tab"""
    
    ROLES = ['organizer', 'executor', 'mediator', 'beneficiary', 
             'affected_party', 'provocateur', 'observer', 'unknown']
    
    FUNCTIONS = ['leader', 'spokesperson', 'expert', 'witness', 
                 'information_source', 'unknown']
    
    POSITIONS = ['state_actor', 'commercial_actor', 'non_commercial_actor', 
                 'international_actor', 'illegal_actor', 'religious_actor', 
                 'individual_actor', 'unknown']
    
    INFO_ROLES = ['information_spreader', 'information_target', 
                  'narrative_creator', 'unknown']
    
    def __init__(self, parent_frame, participation_manager, db):
        """Initialize Participation tab"""
        self.participation_manager = participation_manager
        self.db = db
        self.parent = parent_frame
        
        self.current_participation_id = None
        self.current_event_id = None
        self.current_actor_id = None
        self.form_vars = {}
        
        self.create_ui()
        logger.info("Participation tab initialized")
    
    def create_ui(self):
        """Create complete UI for participation tab"""
        
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== TOP BUTTONS =====
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="🔍 Select Event", 
                  command=self.select_event).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="👤 Select Actor", 
                  command=self.select_actor).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🗑️ Clear Form", 
                  command=self.clear_form).pack(side=tk.LEFT, padx=5)
        
        # ===== FORM FIELDS =====
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame
        canvas = tk.Canvas(form_frame, bg='white')
        scrollbar = ttk.Scrollbar(form_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)


        # Update canvas width when window size changes
        def on_frame_configure(event):
            canvas.itemconfig(1, width=event.width)
        form_frame.bind("<Configure>", on_frame_configure)

    
        row = 0
        
        # ===== EVENT & ACTOR DISPLAY =====
        ttk.Label(scrollable_frame, text="Event ID:", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.form_vars['event_id_display'] = ttk.Label(scrollable_frame, text="None", relief=tk.SUNKEN)
        self.form_vars['event_id_display'].grid(row=row, column=1, sticky=tk.EW, padx=10, ipady=5)
        row += 1
        
        ttk.Label(scrollable_frame, text="Event Title:", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.form_vars['event_title_display'] = ttk.Label(scrollable_frame, text="None", relief=tk.SUNKEN)
        self.form_vars['event_title_display'].grid(row=row, column=1, sticky=tk.EW, padx=10, ipady=5)
        row += 1
        
        ttk.Label(scrollable_frame, text="Actor ID:", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.form_vars['actor_id_display'] = ttk.Label(scrollable_frame, text="None", relief=tk.SUNKEN)
        self.form_vars['actor_id_display'].grid(row=row, column=1, sticky=tk.EW, padx=10, ipady=5)
        row += 1
        
        ttk.Label(scrollable_frame, text="Actor Name:", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.form_vars['actor_name_display'] = ttk.Label(scrollable_frame, text="None", relief=tk.SUNKEN)
        self.form_vars['actor_name_display'].grid(row=row, column=1, sticky=tk.EW, padx=10, ipady=5)
        row += 1
        
        # ===== CLASSIFICATION FIELDS =====
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Participation Role:", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.form_vars['participation_role'] = tk.StringVar(value='unknown')
        ttk.Combobox(scrollable_frame, textvariable=self.form_vars['participation_role'], 
                    values=self.ROLES, state='readonly', width=30).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Participation Function:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.form_vars['participation_function'] = tk.StringVar(value='unknown')
        ttk.Combobox(scrollable_frame, textvariable=self.form_vars['participation_function'], 
                    values=self.FUNCTIONS, state='readonly', width=30).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Participation Position:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.form_vars['participation_position'] = tk.StringVar(value='unknown')
        ttk.Combobox(scrollable_frame, textvariable=self.form_vars['participation_position'], 
                    values=self.POSITIONS, state='readonly', width=30).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Information Role:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.form_vars['information_role'] = tk.StringVar(value='unknown')
        ttk.Combobox(scrollable_frame, textvariable=self.form_vars['information_role'], 
                    values=self.INFO_ROLES, state='readonly', width=30).grid(row=row, column=1, sticky=tk.W, padx=10)
        row += 1
        
        # ===== DESCRIPTION & SOURCE =====
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)
        row += 1
        
        ttk.Label(scrollable_frame, text="Description:", font=('', 14, 'bold')).grid(row=row, column=0, sticky=tk.NW, pady=5)
        text_description = scrolledtext.ScrolledText(scrollable_frame, height=4, width=80, wrap=tk.WORD)
        text_description.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        self.form_vars['description_widget'] = text_description
        row += 1
        
        ttk.Label(scrollable_frame, text="Source:").grid(row=row, column=0, sticky=tk.NW, pady=5)
        text_source = scrolledtext.ScrolledText(scrollable_frame, height=4, width=80, wrap=tk.WORD)
        text_source.grid(row=row, column=1, sticky=tk.EW, padx=10, pady=5)
        self.form_vars['source_widget'] = text_source
        row += 1
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ===== ACTION BUTTONS =====
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="💾 Save Participation", 
                  command=self.save_participation).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="❌ Cancel", 
                  command=self.clear_form).pack(side=tk.LEFT, padx=5)
        
        # Configure column weights
        scrollable_frame.columnconfigure(1, weight=1)
    
    def select_event(self):
        """Open event search modal"""
        modal = ParticipationSearchEventsModal(self.parent, self.db)
        self.parent.wait_window(modal)
        
        if modal.selected_event:
            self.current_event_id = modal.selected_event['event_id']
            self.form_vars['event_id_display'].config(text=str(self.current_event_id))
            self.form_vars['event_title_display'].config(text=modal.selected_event['event_title_ru'])
    
    def select_actor(self):
        """Open actor search modal"""
        modal = SearchActorsModal(self.parent, self.db)
        self.parent.wait_window(modal)
        
        if modal.selected_actor:
            self.current_actor_id = modal.selected_actor['actor_id']
            self.form_vars['actor_id_display'].config(text=self.current_actor_id[:12] + "...")
            self.form_vars['actor_name_display'].config(text=modal.selected_actor['actor_name'])
    
    def get_form_data(self) -> Dict:
        """Collect form data into dictionary"""
        return {
            'event_id': self.current_event_id,
            'actor_id': self.current_actor_id,
            'participation_role': self.form_vars['participation_role'].get(),
            'participation_function': self.form_vars['participation_function'].get(),
            'participation_position': self.form_vars['participation_position'].get(),
            'information_role': self.form_vars['information_role'].get(),
            'description': self.form_vars['description_widget'].get('1.0', tk.END).strip(),
            'source': self.form_vars['source_widget'].get('1.0', tk.END).strip()
        }
    
    def save_participation(self):
        """Save participation to database"""
        try:
            participation_data = self.get_form_data()
            
            if self.current_participation_id:
                # Update existing
                self.participation_manager.update_participation(
                    self.current_participation_id, participation_data)
                messagebox.showinfo("Success", 
                    f"Participation {self.current_participation_id} updated successfully")
                logger.info(f"Updated participation {self.current_participation_id}")
            else:
                # Create new
                participation_id = self.participation_manager.create_participation(participation_data)
                messagebox.showinfo("Success", 
                    f"Participation created successfully (ID: {participation_id})")
                logger.info(f"Created participation {participation_id}")
                self.current_participation_id = participation_id
            
            self.clear_form()
        
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
        except Exception as e:
            logger.error(f"Error saving participation: {e}")
            messagebox.showerror("Error", f"Error saving participation: {str(e)}")
    
    def clear_form(self):
        """Clear all form fields"""
        self.current_participation_id = None
        self.current_event_id = None
        self.current_actor_id = None
        
        self.form_vars['event_id_display'].config(text="None")
        self.form_vars['event_title_display'].config(text="None")
        self.form_vars['actor_id_display'].config(text="None")
        self.form_vars['actor_name_display'].config(text="None")
        
        self.form_vars['participation_role'].set('unknown')
        self.form_vars['participation_function'].set('unknown')
        self.form_vars['participation_position'].set('unknown')
        self.form_vars['information_role'].set('unknown')
        
        self.form_vars['description_widget'].delete('1.0', tk.END)
        self.form_vars['source_widget'].delete('1.0', tk.END)


class NetworkAnalyticsApplication:
    """Main GUI application for network analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🔗 Network Analytics Pro - Actor-Event Database")
        self.root.geometry("1600x950")
        self.root.configure(bg="#ffffff")
        
        # Initialize components
        self.db = NetworkDatabase(DB_PATH)
        self.actor_manager = ActorManager(self.db)
        self.name_lookup = NameLookupEngine(self.db)
        self.edge_manager = EdgeManager(self.db)
        self.event_manager = EventManager(self.db)
        self.participation_manager = ParticipationManager(self.db)
        self.history_manager = SearchHistoryManager()
        self.session = SessionContext()
        
        # Setup UI
        self.setup_styles()
        self.create_menu()
        self.create_main_layout()

        # Initialize bio tab reference (will be set in create_actors_tab)
        self.actors_bio_tab = None

        
        logger.info("Application initialized")
    
    # def setup_styles(self):
    #     """Configure Tkinter styles"""
    #     style = ttk.Style()
    #     style.theme_use('clam')      
    #     # Define custom colors
    #     style.configure('TNotebook', background="#f0f0f0")
    #     style.configure('TFrame', background="#f0f0f0")
    #     style.configure('TLabel', background="#f0f0f0")
    #     style.configure('Unsaved.TLabel', background="#fff9e6")
    #     style.configure('Saved.TLabel', background="#e6f7e6")

    def setup_styles(self):
        """Configure minimalistic flat design"""
        style = ttk.Style()
        style.theme_use('alt')
        # Minimal color palette
        style.configure('.', background='#ffffff')
        style.configure('TNotebook', background='#f8f9fa')
        style.configure('TNotebook.Tab', 
                    background='#e9ecef',
                    foreground='#495057',
                    padding=[20, 8],
                    font=('Arial', 12))
        style.configure('TFrame', background='#ffffff')
        style.configure('TLabel', background='#ffffff', font=('Arial', 12))
        style.configure('Unsaved.TLabel', background='#fff3bf')
        style.configure('Saved.TLabel', background='#d3f9d8')

    def create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="📁 File", menu=file_menu)
        file_menu.add_command(label="Export to JSON", command=self.export_json)
        file_menu.add_command(label="Export to CSV", command=self.export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="🔧 Tools", menu=tools_menu)
        tools_menu.add_command(label="Database Statistics", command=self.show_statistics)
        tools_menu.add_command(label="Verify Data Integrity", command=self.verify_integrity)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="❓ Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_layout(self):
        """Create main application layout"""
        
        # Top status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=9, pady=9)
            
        # Bind tab change event
        self.notebook.bind('<<NotebookTabChanged>>', self.on_notebook_tab_changed)
        
        # Create tabs
        self.create_actor_id_tab()
        self.create_actors_tab()
        self.create_edges_tab()
        self.create_events_tab()
        self.create_participation_tab()
        self.create_statistics_tab()


    def on_notebook_tab_changed(self, event=None):
        """Handle notebook tab changes - auto-load actor data"""
        selected_tab_index = self.notebook.index(self.notebook.select())
        selected_tab_text = self.notebook.tab(selected_tab_index)['text']
        
        # If switching to Actors tab, load active actor
        if "Actors Biographical" in selected_tab_text:
            if self.actors_bio_tab and self.session.active_actor_id:
                self.actors_bio_tab.load_active_actor()
    
    def create_actor_id_tab(self):
        """Tab 1: Create Actor ID"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="👤 Create Actor ID")
        
        # Main form
        form_frame = ttk.LabelFrame(frame, text="Create New Actor Identity", padding=15)
        form_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Nationality
        ttk.Label(form_frame, text="Nationality:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.actor_id_nationality = ttk.Combobox(
            form_frame,
            values=['ru', 'zh_prc', 'zh_hk', 'zh_sg', 'zh_tw', 'other'],
            width=30,
            state='readonly'
        )
        self.actor_id_nationality.grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        
        # Name
        ttk.Label(form_frame, text="Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.actor_id_name = ttk.Entry(form_frame, width=40)
        self.actor_id_name.grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)
        
        # Born (MM-YYYY)
        ttk.Label(form_frame, text="Birth (MM-YYYY):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.actor_id_born = ttk.Entry(form_frame, width=40)
        self.actor_id_born.grid(row=2, column=1, sticky=tk.EW, padx=10, pady=5)
        self.actor_id_born.insert(0, "00-1900")
        
        # Source
        ttk.Label(form_frame, text="Source:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.actor_id_source = ttk.Entry(form_frame, width=40)
        self.actor_id_source.grid(row=3, column=1, sticky=tk.EW, padx=10, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=15)
        
        ttk.Button(button_frame, text="✚ Create Actor ID", command=self.create_actor_id_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Clear", command=self.clear_actor_id_form).pack(side=tk.LEFT, padx=5)
        
        # Result display
        result_frame = ttk.LabelFrame(frame, text="Created Actor IDs", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.actor_id_result_text = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.actor_id_result_text.pack(fill=tk.BOTH, expand=True)
    
    def create_actor_id_action(self):
        """Handle create actor ID action"""
        try:
            nationality = self.actor_id_nationality.get()
            name = self.actor_id_name.get()
            born = self.actor_id_born.get()
            source = self.actor_id_source.get() or "Manual entry"
            
            if not nationality or not name:
                messagebox.showwarning("Validation Error", "Nationality and name are required")
                return
            
            actor_id = self.actor_manager.create_actor_id(nationality, name, born, source)
            self.session.active_actor_id = actor_id
            
            # Display result
            info = self.actor_manager.get_actor_id_info(actor_id)
            result_text = f"""
✓ Actor ID Created Successfully!

Actor ID: {actor_id}
Nationality: {info['nationality']}
Name: {name}
Birth: {born}
Source: {source}
Created: {info['created_date']}

This actor ID is now ACTIVE for data entry.
"""
            self.actor_id_result_text.insert(tk.END, result_text + "\n" + "="*70 + "\n\n")
            self.actor_id_result_text.see(tk.END)
            
            self.update_status(f"✓ Actor created: {name}")
            self.clear_actor_id_form()
            
        except Exception as e:
            logger.error(f"Error creating actor ID: {e}")
            messagebox.showerror("Error", str(e))
    
    def clear_actor_id_form(self):
        """Clear actor ID form"""
        self.actor_id_nationality.set('')
        self.actor_id_name.delete(0, tk.END)
        self.actor_id_born.delete(0, tk.END)
        self.actor_id_born.insert(0, "00-1900")
        self.actor_id_source.delete(0, tk.END)
    

    def create_actors_tab(self):
        """Tab 2: Biographical Information"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📋 Actors Biographical")
        
        # Initialize the biographical tab
        self.actors_bio_tab = ActorsBiographicalTab(
            parent_frame=frame,
            actor_manager=self.actor_manager,
            session=self.session,
            db=self.db
        )
    

    def create_edges_tab(self):
        """Create Tab 3: Edges/Relationships tab"""
        edges_frame = ttk.Frame(self.notebook)
        self.notebook.add(edges_frame, text="🔗 Edges/Relationships")
        
        self.edges_tab = EdgesTab(
            parent_frame=edges_frame,
            edge_manager=self.edge_manager,
            actor_manager=self.actor_manager,
            name_lookup=self.name_lookup,
            session=self.session,
            db=self.db
        )
        
        logger.info("Edges tab created")


    def create_events_tab(self):
        """Create Tab 4: Events tab"""
        events_frame = ttk.Frame(self.notebook)
        self.notebook.add(events_frame, text="📋 Events")
        
        self.events_tab = EventsTab(
            parent_frame=events_frame,
            event_manager=self.event_manager,
            history_manager=self.history_manager,
            db=self.db
    )
    logger.info("Events tab created")
    
    

    def create_participation_tab(self):
        """Tab 5: Create Participation tab"""
        participation_frame = ttk.Frame(self.notebook)
        self.notebook.add(participation_frame, text="🤝 Participation")
        
        self.participation_tab = ParticipationTab(
            parent_frame=participation_frame,
            participation_manager=self.participation_manager,
            db=self.db
        )
    
    logger.info("Participation tab created")
    
    def create_statistics_tab(self):
        """Tab 6: Statistics"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Statistics")
        
        ttk.Button(frame, text="Compute Statistics", command=self.compute_statistics).pack(pady=10)
        
        self.stats_text = scrolledtext.ScrolledText(frame, height=25)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def compute_statistics(self):
        """Compute and display database statistics"""
        try:
            self.stats_text.delete("1.0", tk.END)
            
            # Get counts
            actors_id_count = self.db.execute_query("SELECT COUNT(*) as cnt FROM actors_id")[0]['cnt']
            actors_records = self.db.execute_query("SELECT COUNT(*) as cnt FROM Actors")[0]['cnt']
            aliases_count = self.db.execute_query("SELECT COUNT(*) as cnt FROM Actor_Aliases")[0]['cnt']
            events_count = self.db.execute_query("SELECT COUNT(*) as cnt FROM Events")[0]['cnt']
            participations = self.db.execute_query("SELECT COUNT(*) as cnt FROM Event_Participation")[0]['cnt']
            
            stats = f"""
╔════════════════════════════════════════════════════════════╗
║           DATABASE STATISTICS                              ║
╚════════════════════════════════════════════════════════════╝

👤 ACTORS:
   • Unique Actor IDs: {actors_id_count}
   • Biographical Records: {actors_records}
   • Name Aliases: {aliases_count}

📅 EVENTS:
   • Total Events: {events_count}
   • Event Participations: {participations}

🔗 RELATIONSHIPS:
   (To be computed)

📊 QUALITY METRICS:
   • Database File: {DB_PATH}
   • Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
            self.stats_text.insert("1.0", stats)
            self.update_status("✓ Statistics computed")
            
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            messagebox.showerror("Error", str(e))
    
    def export_json(self):
        """Export data to JSON"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not filename:
                return
            
            # Collect all data
            export_data = {
                'actors_id': [dict(r) for r in self.db.execute_query('SELECT * FROM actors_id')],
                'actors': [dict(r) for r in self.db.execute_query('SELECT * FROM Actors')],
                'aliases': [dict(r) for r in self.db.execute_query('SELECT * FROM Actor_Aliases')],
                'events': [dict(r) for r in self.db.execute_query('SELECT * FROM Events')],
                'participations': [dict(r) for r in self.db.execute_query('SELECT * FROM Event_Participation')],
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            messagebox.showinfo("Success", f"Data exported to {filename}")
            self.update_status(f"✓ Exported to {filename}")
            logger.info(f"Data exported to {filename}")
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            messagebox.showerror("Error", str(e))
    
    def export_csv(self):
        """Export actors to CSV"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not filename:
                return
            
            import csv
            
            actors = self.db.execute_query('''
                SELECT a.*, aid.actor_id, aid.nationality, aid.name_ru, aid.name_zh_prc, aid.name_en
                FROM Actors a
                JOIN actors_id aid ON a.actor_id = aid.actor_id
            ''')
            
            if not actors:
                messagebox.showinfo("Info", "No actors to export")
                return
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=dict(actors[0]).keys())
                writer.writeheader()
                for row in actors:
                    writer.writerow(dict(row))
            
            messagebox.showinfo("Success", f"Actors exported to {filename}")
            logger.info(f"Actors exported to {filename}")
            
        except Exception as e:
            logger.error(f"CSV export error: {e}")
            messagebox.showerror("Error", str(e))
    
    def show_statistics(self):
        """Show database statistics"""
        self.notebook.select(self.notebook.index("end")-1)  # Select last tab
        self.compute_statistics()
    
    def verify_integrity(self):
        """Verify database integrity"""
        try:
            msg = "Database integrity verification:\n"
            
            # Check FK constraints
            missing_refs = self.db.execute_query('''
                SELECT a.actor_id FROM Actors a
                WHERE NOT EXISTS (SELECT 1 FROM actors_id WHERE actor_id = a.actor_id)
            ''')
            
            if missing_refs:
                msg += f"⚠️ {len(missing_refs)} orphaned Actors records\n"
            else:
                msg += "✓ All Actors have valid references\n"
            
            messagebox.showinfo("Integrity Check", msg)
            logger.info("Integrity verification completed")
            
        except Exception as e:
            logger.error(f"Integrity check error: {e}")
            messagebox.showerror("Error", str(e))
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", """
🔗 Network Analytics Pro v1.0

Actor-Event Network Database

Analyzing hidden networks

Features:
                            
• Multilingual name support
• Temporal actor tracking
• Event management and analysis
• Intelligent name lookup
• Comprehensive audit trails

Database: SQLite3
GUI: Tkinter
Python 3.8+

© 2025 Zonengeist
        """)
    
    def update_status(self, message: str):
        """Update status bar"""
        indicator = ""
        if self.session.active_actor_id:
            indicator = f"Active: {self.session.active_actor_id[:8]}... | "
        
        unsaved = " (Unsaved)" if self.session.has_unsaved() else ""
        self.status_label.config(text=f"{indicator}{message}{unsaved}")
    
    def on_close(self):
        """Handle window close"""
        if self.session.has_unsaved():
            if messagebox.askyesno("Unsaved Changes", "You have unsaved changes. Exit anyway?"):
                self.db.disconnect()
                self.root.destroy()
                logger.info("Application closed with unsaved changes")
        else:
            self.db.disconnect()
            self.root.destroy()
            logger.info("Application closed normally")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkAnalyticsApplication(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
