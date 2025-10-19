# =====================================================
# FILE: migrations/versions/001_initial_schema.py
# LOCATION: migrations/versions/001_initial_schema.py
# Initial database schema migration
# =====================================================

"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2025-01-16 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('email_verified', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('last_login', sa.DateTime)
    )
    op.create_index('idx_users_email', 'users', ['email'])
    
    # User profiles
    op.create_table(
        'user_profiles',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('display_name', sa.String(255)),
        sa.Column('avatar_url', sa.String(500)),
        sa.Column('grade_level', sa.String(50)),
        sa.Column('school_id', postgresql.UUID(as_uuid=True)),
        sa.Column('accessibility_needs', postgresql.JSON, server_default='{}'),
        sa.Column('language_preference', sa.String(10), server_default='en'),
        sa.Column('timezone', sa.String(50), server_default='UTC'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # Learning profiles
    op.create_table(
        'learning_profiles',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('modality_preferences', postgresql.JSON),
        sa.Column('pace_preference', sa.String(20), server_default='medium'),
        sa.Column('detail_level', sa.String(20), server_default='medium'),
        sa.Column('tts_rate', sa.Numeric(3, 2), server_default='1.00'),
        sa.Column('hint_frequency', sa.String(20), server_default='moderate'),
        sa.Column('preferred_explanation_style', sa.String(50)),
        sa.Column('reading_level', sa.Numeric(3, 1)),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # Study sessions
    op.create_table(
        'study_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True)),
        sa.Column('assignment_id', postgresql.UUID(as_uuid=True)),
        sa.Column('session_type', sa.String(50), nullable=False),
        sa.Column('started_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('ended_at', sa.DateTime),
        sa.Column('total_duration_seconds', sa.Integer),
        sa.Column('topic', sa.String(255)),
        sa.Column('metadata', postgresql.JSON, server_default='{}')
    )
    op.create_index('idx_study_sessions_user', 'study_sessions', ['user_id'])
    
    # User progress (gamification)
    op.create_table(
        'user_progress',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('total_xp', sa.Integer, server_default='0'),
        sa.Column('level', sa.Integer, server_default='1'),
        sa.Column('current_streak_days', sa.Integer, server_default='0'),
        sa.Column('longest_streak_days', sa.Integer, server_default='0'),
        sa.Column('last_activity_date', sa.DateTime),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now())
    )


def downgrade() -> None:
    op.drop_table('user_progress')
    op.drop_index('idx_study_sessions_user')
    op.drop_table('study_sessions')
    op.drop_table('learning_profiles')
    op.drop_table('user_profiles')
    op.drop_index('idx_users_email')
    op.drop_table('users')
