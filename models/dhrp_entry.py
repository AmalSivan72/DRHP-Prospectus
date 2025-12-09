from models import db

class DhrpEntry(db.Model):
    __tablename__ = 'dhrp_entries'  # ✅ Explicit table name

    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(100))
    bse_code = db.Column(db.String(30))  # Increased length to avoid DataError
    upload_date = db.Column(db.String(20))
    uploader_name = db.Column(db.String(100))
    promoter = db.Column(db.String(100))
    pdf_filename = db.Column(db.String(200), unique=True)
    status = db.Column(db.String(50))
    toc_verified = db.Column(db.Boolean, default=False)


class ProcessingStatus(db.Model):
    __tablename__ = 'processing_status'
    id = db.Column(db.Integer, primary_key=True)
    dhrp_id = db.Column(db.Integer, db.ForeignKey('dhrp_entries.id'), unique=True)
    processing_stage = db.Column(db.String(200))
    updated_at = db.Column(db.String(50))

    dhrp = db.relationship('DhrpEntry', backref=db.backref('processing_status', uselist=False))

class TocSection(db.Model):
    __tablename__ = 'toc_sections'
    id = db.Column(db.Integer, primary_key=True)
    dhrp_id = db.Column(db.Integer, db.ForeignKey('dhrp_entries.id'))
    title = db.Column(db.String(200))
    page = db.Column(db.Integer)
    subsection_title = db.Column(db.String(200), nullable=True)
    subsection_page = db.Column(db.Integer, nullable=True)

    dhrp = db.relationship('DhrpEntry', backref='toc_sections')

class RiskSummary(db.Model):
    __tablename__ = 'risk_summaries'
    id = db.Column(db.Integer, primary_key=True)
    dhrp_id = db.Column(db.Integer, db.ForeignKey('dhrp_entries.id'), unique=True)
    risk_text = db.Column(db.Text)
    summary_bullets = db.Column(db.Text)  # Store as JSON string

    dhrp = db.relationship('DhrpEntry', backref=db.backref('risk_summary', uselist=False))

class QaResult(db.Model):
    __tablename__ = 'qa_results'
    id = db.Column(db.Integer, primary_key=True)
    dhrp_id = db.Column(db.Integer, db.ForeignKey('dhrp_entries.id'))
    question = db.Column(db.Text)
    answer = db.Column(db.Text)

    dhrp = db.relationship('DhrpEntry', backref='qa_results')




class FundEvaluation(db.Model):
    __tablename__ = 'fund_evaluations'

    id = db.Column(db.Integer, primary_key=True)
    dhrp_id = db.Column(db.Integer, db.ForeignKey('dhrp_entries.id'))
    base = db.Column(db.String(255), nullable=False)

    # Store the original question / data field
    data_field = db.Column(db.String(255), nullable=False)

    # Store the extracted answer directly
    answer = db.Column(db.Text, nullable=True)

    # Review status: "approved", "rejected", or "pending"
    review_status = db.Column(db.String(50), default="pending")

    # Relationship back to DhrpEntry
    dhrp = db.relationship('DhrpEntry', backref='fund_evaluations')

    def __repr__(self):
        return (f"<FundEvaluation(base={self.base}, field={self.data_field}, "
                f"answer={self.answer}, status={self.review_status})>")

