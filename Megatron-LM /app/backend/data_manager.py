import json
import yaml
import pandas as pd
from typing import Dict, List, Any
import sqlite3
import os

class DataManager:
    def __init__(self):
        self.knowledge_base = {}
        self.context_data = {}
        self.load_data_sources()
    
    def load_data_sources(self):
        """Cargar diferentes fuentes de datos"""
        self._load_knowledge_base()
        self._load_faqs()
        self._load_context_rules()
        self._init_database()
    
    def _load_knowledge_base(self):
        """Cargar base de conocimiento desde JSON"""
        try:
            with open('data/knowledge_base.json', 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        except FileNotFoundError:
            self.knowledge_base = {
                "company_info": {
                    "name": "Mi Empresa",
                    "services": ["consultoría", "desarrollo", "soporte"],
                    "contact": "info@empresa.com"
                },
                "product_info": {},
                "technical_data": {}
            }
    
    def _load_faqs(self):
        """Cargar preguntas frecuentes"""
        try:
            with open('data/faqs.json', 'r', encoding='utf-8') as f:
                self.faqs = json.load(f)
        except FileNotFoundError:
            self.faqs = {
                "preguntas": [
                    {
                        "pregunta": "¿Qué servicios ofrecen?",
                        "respuesta": "Ofrecemos servicios de consultoría, desarrollo de software y soporte técnico."
                    },
                    {
                        "pregunta": "¿Cómo puedo contactarlos?",
                        "respuesta": "Puedes contactarnos en info@empresa.com o por nuestro formulario web."
                    }
                ]
            }
    
    def _load_context_rules(self):
        """Cargar reglas de contexto"""
        try:
            with open('data/context_rules.yaml', 'r', encoding='utf-8') as f:
                self.context_rules = yaml.safe_load(f)
        except FileNotFoundError:
            self.context_rules = {
                "contexts": {
                    "saludo": {
                        "patterns": ["hola", "buenos días", "buenas tardes"],
                        "response_templates": [
                            "¡Hola! ¿En qué puedo ayudarte hoy?",
                            "¡Buenos días! ¿Cómo puedo asistirte?"
                        ]
                    },
                    "despedida": {
                        "patterns": ["adiós", "chao", "hasta luego"],
                        "response_templates": [
                            "¡Hasta luego! Que tengas un buen día.",
                            "¡Chao! Fue un gusto ayudarte."
                        ]
                    }
                }
            }
    
    def _init_database(self):
        """Inicializar base de datos SQLite para datos dinámicos"""
        os.makedirs('data', exist_ok=True)
        self.conn = sqlite3.connect('data/chat_data.db', check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        """Crear tablas de la base de datos"""
        cursor = self.conn.cursor()
        
        # Tabla de conversaciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message TEXT,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT
            )
        ''')
        
        # Tabla de conocimiento específico
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS custom_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                question TEXT,
                answer TEXT,
                category TEXT,
                confidence REAL DEFAULT 1.0
            )
        ''')
        
        self.conn.commit()
    
    def get_contextual_response(self, message: str, conversation_history: List[Dict]) -> Dict:
        """Obtener respuesta contextual basada en datos"""
        message_lower = message.lower()
        
        # 1. Buscar en FAQs
        faq_response = self._search_faqs(message_lower)
        if faq_response:
            return faq_response
        
        # 2. Buscar en conocimiento personalizado
        custom_response = self._search_custom_knowledge(message_lower)
        if custom_response:
            return custom_response
        
        # 3. Aplicar reglas de contexto
        context_response = self._apply_context_rules(message_lower)
        if context_response:
            return context_response
        
        # 4. Respuesta por defecto
        return {
            "type": "default",
            "response": "No tengo información específica sobre eso. ¿Podrías reformular tu pregunta?",
            "confidence": 0.1,
            "sources": []
        }
    
    def _search_faqs(self, message: str) -> Dict:
        """Buscar en preguntas frecuentes"""
        for faq in self.faqs.get("preguntas", []):
            if any(keyword in message for keyword in faq["pregunta"].lower().split()):
                return {
                    "type": "faq",
                    "response": faq["respuesta"],
                    "confidence": 0.9,
                    "sources": ["faqs"]
                }
        return None
    
    def _search_custom_knowledge(self, message: str) -> Dict:
        """Buscar en conocimiento personalizado de la base de datos"""
        cursor = self.conn.cursor()
        
        # Buscar por palabras clave en preguntas
        keywords = message.split()
        placeholders = ','.join('?' * len(keywords))
        
        query = f'''
            SELECT question, answer, confidence 
            FROM custom_knowledge 
            WHERE question LIKE '%' || ? || '%'
            OR answer LIKE '%' || ? || '%'
            ORDER BY confidence DESC
            LIMIT 1
        '''
        
        cursor.execute(query, (keywords[0], keywords[0]))
        result = cursor.fetchone()
        
        if result:
            return {
                "type": "custom_knowledge",
                "response": result[1],
                "confidence": result[2],
                "sources": ["database"]
            }
        
        return None
    
    def _apply_context_rules(self, message: str) -> Dict:
        """Aplicar reglas de contexto"""
        for context_name, context_data in self.context_rules.get("contexts", {}).items():
            for pattern in context_data["patterns"]:
                if pattern in message:
                    import random
                    response = random.choice(context_data["response_templates"])
                    return {
                        "type": "context",
                        "response": response,
                        "confidence": 0.8,
                        "sources": ["context_rules"]
                    }
        return None
    
    def add_custom_knowledge(self, topic: str, question: str, answer: str, category: str = "general"):
        """Agregar nuevo conocimiento a la base de datos"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO custom_knowledge (topic, question, answer, category)
            VALUES (?, ?, ?, ?)
        ''', (topic, question, answer, category))
        self.conn.commit()
    
    def save_conversation(self, user_id: str, message: str, response: str, context: str = ""):
        """Guardar conversación en la base de datos"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_id, message, response, context)
            VALUES (?, ?, ?, ?)
        ''', (user_id, message, response, context))
        self.conn.commit()
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Obtener historial de conversación"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT message, response, timestamp 
            FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        return [
            {"message": row[0], "response": row[1], "timestamp": row[2]}
            for row in cursor.fetchall()
                             ]
