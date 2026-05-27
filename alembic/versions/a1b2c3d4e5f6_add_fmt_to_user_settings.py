"""add fmt to user_settings

Revision ID: a1b2c3d4e5f6
Revises: 2aacc14dfb97
Create Date: 2026-05-27 18:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "2aacc14dfb97"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_settings",
        sa.Column("fmt", sa.String(length=20), nullable=False, server_default="html"),
    )


def downgrade() -> None:
    op.drop_column("user_settings", "fmt")
