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
            CREATE TABLE IF NOT EXISTS Events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                event_name_ru TEXT NOT NULL,
                event_name_zh TEXT,
                event_name_en TEXT,
                
                event_date DATE NOT NULL,
                event_date_precision TEXT DEFAULT 'day' CHECK(event_date_precision IN ('day', 'month', 'year', 'unknown')),
                
                description_ru TEXT,
                description_zh TEXT,
                description_en TEXT,
                
                event_type TEXT,
                participants_count INTEGER,
                
                sources TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_date ON Events(event_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON Events(event_type)')
        
        # TABLE 5: Event_Participation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Event_Participation (
                participation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                event_id INTEGER NOT NULL REFERENCES Events(event_id),
                actor_id TEXT NOT NULL REFERENCES actors_id(actor_id),
                
                role TEXT,
                participation_type TEXT CHECK(participation_type IN ('organizer', 'speaker', 'attendee', 'victim', 'unknown')),
                
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(event_id, actor_id)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_participation_event ON Event_Participation(event_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_participation_actor ON Event_Participation(actor_id)')
        
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
        """Tab 3: Relationships/Edges"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔗 Edges/Relationships")
        
        ttk.Label(frame, text="Edge Management (Implementation in progress)").pack(pady=20)
    
    def create_events_tab(self):
        """Tab 4: Events"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📅 Events")
        
        ttk.Label(frame, text="Event Management (Implementation in progress)").pack(pady=20)
    
    def create_participation_tab(self):
        """Tab 5: Event Participation"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="👥 Participation")
        
        ttk.Label(frame, text="Event Participation (Implementation in progress)").pack(pady=20)
    
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
