from sqlalchemy import text, select, insert, and_, distinct, case, or_, func, Column, Integer, BigInteger, SmallInteger, VARCHAR, NVARCHAR, DECIMAL
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql.expression import cast
from time import time
import csv
import asyncio
import nest_asyncio
from itertools import islice
import pandas as pd

nest_asyncio.apply()

Base = declarative_base()

class fu_group(Base):
    __tablename__= 'fu_group'
    id = Column(BigInteger, primary_key=True)
    pterm_id = Column(Integer, index=True)
    pterm_name = Column(VARCHAR, index=True)
    is_virtual = Column(SmallInteger, index=True)
    psubject_name = Column(VARCHAR, index=True)
    psubject_code = Column(VARCHAR, index=True)

class fu_group_member(Base):
    __tablename__ = 'fu_group_member'
    id = Column(BigInteger, primary_key=True)
    groupid = Column(Integer, index=True)
    member_login = Column(VARCHAR, index=True)
    user_code = Column(VARCHAR, index=True)
    loai = Column(VARCHAR, index=True)

class fu_activity(Base):
    __tablename__ = 'fu_activity'
    id = Column(BigInteger, primary_key=True)
    groupid = Column(Integer, index=True)
    course_slot = Column(SmallInteger, index=True)
    term_id = Column(Integer, index=True)

class fu_attendance(Base):
    __tablename__ = 'fu_attendance'
    id = Column(BigInteger, primary_key=True)
    activity_id = Column(BigInteger, index=True)
    user_login = Column(VARCHAR, index=True)
    groupid = Column(Integer, index=True)
    val = Column(SmallInteger, index=True)

class fu_user(Base):
    __tablename__ = 'fu_user'
    id = Column(BigInteger, primary_key=True)
    user_login = Column(VARCHAR, index=True)
    user_code = Column(VARCHAR, index=True)

class fz_list_student(Base):
    __tablename__ = 'fz_list_student'
    id = Column(BigInteger, primary_key=True)
    student_code = Column(VARCHAR, index=True)
    semester = Column(Integer, index=True)
    term_id = Column(Integer, index=True)
    status = Column(VARCHAR, index=True)
    major_id = Column(Integer, index=True)
    campus_code = Column(VARCHAR, index=True)

class sub1(Base):
    __tablename__ = 'total_activity'
    id = Column(BigInteger, primary_key=True)
    term_name = Column(VARCHAR(200), nullable=False, index=True)
    term_id = Column(Integer, nullable=False, index=True)
    member_login = Column(VARCHAR(200), nullable=False, index=True)
    member_code = Column(VARCHAR(200), nullable=False, index=True)
    total_act = Column(Integer, nullable=False, index=True)
    group_id = Column(Integer, nullable=False, index=True)

class sub2(Base):
    __tablename__ = 'total_attendance'
    id = Column(BigInteger, primary_key=True)
    attendance_total = Column(Integer, nullable=False, index=True)
    term_name = Column(VARCHAR(200), nullable=False, index=True)
    term_id = Column(Integer, nullable=False, index=True)
    user_login = Column(VARCHAR(200), nullable=False, index=True)
    user_code = Column(VARCHAR(200), nullable=False, index=True)

class sub3(Base):
    __tablename__ = 'total_user_semester'
    id = Column(BigInteger, primary_key=True)
    user_login = Column(VARCHAR(200), nullable=False, index=True)
    user_code = Column(VARCHAR(200), nullable=False, index=True)
    semester = Column(Integer, nullable=False, index=True)
    term_id = Column(Integer, nullable=False, index=True)
    group_id = Column(Integer, nullable=False, index=True)

class t7_course_result(Base):
    __tablename__ = 't7_course_result'
    id = Column(BigInteger, primary_key=True)
    groupid = Column(BigInteger, index=True)
    student_login = Column(VARCHAR, index=True)
    student_code = Column(VARCHAR, index=True)
    val = Column(VARCHAR, index=True)
    grade = Column(DECIMAL(10, 1), index=True)
    psubject_code = Column(VARCHAR(25), index=True)
    skill_code = Column(VARCHAR, index=True)
    number_of_credit = Column(Integer, index=True)
    term_id = Column(Integer, index=True)

class fu_subject(Base):
    __tablename__ = 'fu_subject'
    id = Column(BigInteger, primary_key=True)
    department_id = Column(Integer, index=True)
    subject_name = Column(NVARCHAR, index=True)
    subject_code = Column(VARCHAR(25), index=True)
    skill_code = Column(VARCHAR, index=True)
    num_of_credit = Column(Integer, index=True)

class fu_department(Base):
    __tablename__ = "fu_department"
    id = Column(BigInteger, primary_key=True)
    department_name = Column(VARCHAR(50), index=True)

class brand(Base):
    __tablename__ = 'brand'
    id = Column(BigInteger, primary_key=True)
    code = Column(VARCHAR(100), index=True)
    major = Column(VARCHAR(100), index=True)
    major_code = Column(VARCHAR(100), index=True)

class mapping_term(Base):
    __tablename__ = 'mapping_term'
    id = Column(Integer, primary_key=True)
    ho_term_id = Column(VARCHAR, index=True)
    ph = Column(VARCHAR, index=True)

async def total_activity(engine, partitioner, mod_number):
    q1 = select(fu_group.id, fu_group.pterm_id, fu_user.user_login, fu_user.user_code, fu_group.pterm_name)\
        .join_from(fu_group_member, fu_group, fu_group.id == fu_group_member.groupid)\
        .join_from(fu_group_member, fu_user, fu_group_member.member_login == fu_user.user_login)\
        .where(
            fu_group.pterm_id >= 24,
            fu_group.is_virtual == 0
            ).subquery()
    q2 = select(fu_group.id.label('groupid'), func.count(fu_activity.id).label('act_total'))\
        .join_from(fu_activity, fu_group, fu_activity.groupid == fu_group.id)\
        .where(
            fu_group.pterm_id >= 24,
            fu_group.is_virtual == 0,
            fu_activity.course_slot < 100
        ).group_by(fu_group.id).subquery()
    q = select(q1.c.pterm_id, q1.c.user_login, q1.c.user_code, func.sum(q2.c.act_total).label('total_act'), q1.c.pterm_name, q1.c.id)\
        .join_from(q1, q2, q1.c.id == q2.c.groupid)\
        .where(q1.c.pterm_id % 11 == mod_number)\
        .group_by(q1.c.pterm_id, q1.c.user_code)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def total_attendance(engine, partitioner, mod_number):
    q1 = select(fu_group.id, fu_group.pterm_id, fu_group.pterm_name)\
        .where(fu_group.pterm_id >= 24,
               fu_group.is_virtual == 0
               ).subquery()
    q = select(func.count(fu_attendance.activity_id).label('total_att'), fu_attendance.user_login, 
               fu_user.user_code, q1.c.pterm_id, q1.c.pterm_name)\
        .join_from(fu_attendance, q1, fu_attendance.groupid == q1.c.id)\
        .join_from(fu_attendance, fu_user, fu_attendance.user_login == fu_user.user_login)\
        .where(fu_attendance.val == 1,
               q1.c.pterm_id % 17 == mod_number)\
        .group_by(fu_attendance.user_login, q1.c.pterm_id)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def check_user_semester(engine, partitioner, mod_number):
    q1 = select(t7_course_result.groupid, t7_course_result.student_login,
                t7_course_result.term_id,
                case((~mapping_term.ph.is_(None), mapping_term.ho_term_id), else_=t7_course_result.term_id).label('fz_term'))\
        .join(mapping_term, t7_course_result.term_id == mapping_term.ph, isouter=True)\
        .where(t7_course_result.term_id >= 24)\
        .subquery()
    
    q2 = select(q1.c.groupid, q1.c.student_login, 
                q1.c.term_id, q1.c.fz_term, fu_user.user_code, fu_user.id)\
        .join_from(q1, fu_user, q1.c.student_login == fu_user.user_login)\
        .subquery()

    q = select(q2.c.student_login, q2.c.user_code, 
               fz_list_student.semester, fz_list_student.term_id,
               q2.c.groupid)\
        .join_from(fz_list_student, q2, and_(q2.c.fz_term == fz_list_student.term_id,
                                             q2.c.user_code == fz_list_student.student_code))\
        .where(
               fz_list_student.semester < 3, 
               fz_list_student.student_code.in_(
                   select(fz_list_student.student_code)\
                    .where(fz_list_student.major_id.in_([1, 3, 6, 12, 13, 14, 15, 16, 17,
                                                        28, 29, 30, 31, 32, 33, 34, 35, 
                                                        50, 51, 52, 53, 54, 71, 72]),
                            fz_list_student.campus_code == 'ph',
                            fz_list_student.semester.in_([0, 1, 2]))\
                    .group_by(fz_list_student.student_code)\
                    .having(func.max(fz_list_student.semester) == 2, func.count(distinct(fz_list_student.semester)) >= 2)
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(fz_list_student.campus_code != 'ph')
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(and_(fz_list_student.semester.in_([0, 1, 2]),
                                                                   fz_list_student.status == 'THO'))
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(or_(fz_list_student.term_id < 26,
                                                                  fz_list_student.term_id == 54))
               ),
               q2.c.id % 13 == mod_number)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def percent_attendance(engine, partitioner, mod_number, semester_range):
    q = select(sub3.term_id, sub1.member_login, sub1.member_code,
               func.sum(sub2.attendance_total), func.sum(sub1.total_act),
               func.sum(sub2.attendance_total) / func.sum(sub1.total_act),
               func.max(sub3.semester))\
        .join_from(sub1, sub2, and_(sub1.term_id == sub2.term_id,
                                    sub1.member_code == sub2.user_code))\
        .join_from(sub1, sub3, and_(sub1.group_id == sub3.group_id,
                                    sub1.member_code == sub3.user_code))\
        .group_by(sub1.member_code)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def process_data(chunk, csv_writer):
    for i in chunk:
        for j in i:
            csv_writer.writerow(j)

async def avg_grade(engine, partitioner, mod_number, semester_range):
    q1 = select(t7_course_result.groupid, t7_course_result.student_login,
                t7_course_result.term_id, t7_course_result.val,
                t7_course_result.grade, t7_course_result.number_of_credit,
                t7_course_result.psubject_code,
                case((~mapping_term.ph.is_(None), mapping_term.ho_term_id), else_=t7_course_result.term_id).label('fz_term'))\
        .join(mapping_term, t7_course_result.term_id == mapping_term.ph, isouter=True)\
        .where(t7_course_result.term_id >= 24)\
        .subquery()
    
    q2 = select(q1.c.groupid, q1.c.student_login, fu_user.user_code,
                q1.c.term_id, q1.c.val, q1.c.grade, q1.c.number_of_credit,
                q1.c.psubject_code, q1.c.fz_term, fu_user.id)\
        .join_from(q1, fu_user, q1.c.student_login == fu_user.user_login)\
        .subquery()

    q = select(q2.c.student_login, fz_list_student.student_code, 
               func.max(fz_list_student.semester), q2.c.term_id, fz_list_student.major_id,
               fz_list_student.campus_code, q2.c.val, 
               func.sum(q2.c.grade * q2.c.number_of_credit),
               func.sum(q2.c.number_of_credit),
               func.sum(q2.c.grade * q2.c.number_of_credit) / func.sum(q2.c.number_of_credit))\
        .join_from(fz_list_student, q2, and_(q2.c.fz_term == fz_list_student.term_id,
                                             q2.c.user_code == fz_list_student.student_code))\
        .where(
               fz_list_student.semester < 3, 
               fz_list_student.student_code.in_(
                   select(fz_list_student.student_code)\
                    .where(fz_list_student.major_id.in_([1, 3, 6, 12, 13, 14, 15, 16, 17,
                                                        28, 29, 30, 31, 32, 33, 34, 35, 
                                                        50, 51, 52, 53, 54, 71, 72]),
                            fz_list_student.campus_code == 'ph',
                            fz_list_student.semester.in_([0, 1, 2]))\
                    .group_by(fz_list_student.student_code)\
                    .having(func.max(fz_list_student.semester) == 2, func.count(distinct(fz_list_student.semester)) >= 2)
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(fz_list_student.campus_code != 'ph')
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(and_(fz_list_student.semester.in_([0, 1, 2]),
                                                                   fz_list_student.status == 'THO'))
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(or_(fz_list_student.term_id < 26,
                                                                  fz_list_student.term_id == 54))
               ))\
        .group_by(fz_list_student.student_code)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
      
async def avg_grade_department(engine, partitioner, mod_number, semester_range):
    pivot_columns = []
    q1 = select(func.distinct(func.left(fu_subject.subject_code, 3)))
    async with engine.connect() as conn:
        result = await conn.execute(q1)
        for code in result:
            code_name = code[0]
            pivot_columns.append(text(f"""
                SUM(CASE WHEN LEFT(fu_subject.subject_code, 3) = '{code_name}' AND t7_course_result.val <> 1 THEN 1 ELSE 0 END) AS fail_{code_name}
            """))
            pivot_columns.append(text(f"""
                ROUND(SUM(CASE WHEN LEFT(fu_subject.subject_code, 3) = '{code_name}' AND t7_course_result.val = 1 
                THEN t7_course_result.grade * fu_subject.num_of_credit ELSE 0 END) 
                    / SUM(CASE WHEN LEFT(fu_subject.subject_code, 3) = '{code_name}' AND t7_course_result.val = 1 
                    THEN fu_subject.num_of_credit END), 2) AS avg_grade_{code_name}
            """))

        q1 = select(fu_group.id, fu_group.pterm_id, fu_group_member.member_login, fu_group.pterm_name)\
            .join_from(fu_group, fu_group_member, fu_group.id == fu_group_member.groupid)\
            .where(
                fu_group.pterm_id >= 24,
                fu_group.is_virtual == 0,
                ).subquery()
        
        q = select(q1.c.pterm_id, q1.c.member_login,
                    *pivot_columns,
                    func.max(fz_list_student.semester),
                    func.count(distinct(fz_list_student.semester)))\
            .join_from(q1, t7_course_result, and_(q1.c.id == t7_course_result.groupid,
                                                q1.c.member_login == t7_course_result.student_login))\
            .join_from(t7_course_result, fu_subject, t7_course_result.psubject_code == fu_subject.subject_code)\
            .join_from(t7_course_result, fz_list_student, and_(t7_course_result.student_code == fz_list_student.student_code,
                                                            q1.c.pterm_id == fz_list_student.term_id))\
            .join_from(fz_list_student, brand, brand.id == fz_list_student.major_id)\
            .where(fz_list_student.semester.in_(semester_range), 
                   ((brand.major.like('Công nghệ thông tin%')) | (brand.major_code == 'IT')),
                   fu_subject.department_id == 28)\
            .group_by(q1.c.member_login)\
            .having(and_(func.max(fz_list_student.semester) == len(semester_range), func.count(distinct(fz_list_student.semester)) == len(semester_range)))

        result = await conn.execute(q)
        return result

async def check_status(engine, partitioner, mod_number):
    q = select(fz_list_student.student_code,
               fz_list_student.status, fz_list_student.semester,
               fz_list_student.major_id, fz_list_student.campus_code,
               fz_list_student.term_id
               )\
        .where(
               fz_list_student.semester < 4, 
               fz_list_student.student_code.in_(
                   select(fz_list_student.student_code)\
                    .where(fz_list_student.major_id.in_([1, 3, 6, 12, 13, 14, 15, 16, 17,
                                                        28, 29, 30, 31, 32, 33, 34, 35, 
                                                        50, 51, 52, 53, 54, 71, 72]),
                            fz_list_student.campus_code == 'ph',
                            fz_list_student.semester.in_([0, 1, 2, 3]))\
                    .group_by(fz_list_student.student_code)\
                    .having(func.max(fz_list_student.semester) == 3, func.count(distinct(fz_list_student.semester)) >= 3)
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(fz_list_student.campus_code != 'ph')
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(fz_list_student.semester.in_([0, 1, 2]),
                                                              fz_list_student.status == 'THO')
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(or_(fz_list_student.term_id < 26,
                                                                  fz_list_student.term_id == 54))
               ))
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def avg_grade_by_subject(engine, partitioner, mod_number, semester_range):
    q1 = select(t7_course_result.groupid, t7_course_result.student_login,
                t7_course_result.term_id, t7_course_result.val,
                t7_course_result.grade, t7_course_result.number_of_credit,
                t7_course_result.psubject_code,
                case((~mapping_term.ph.is_(None), mapping_term.ho_term_id), else_=t7_course_result.term_id).label('fz_term'))\
        .join(mapping_term, t7_course_result.term_id == mapping_term.ph, isouter=True)\
        .where(t7_course_result.term_id >= 24)\
        .subquery()
    
    q2 = select(q1.c.groupid, q1.c.student_login, fu_user.user_code,
                q1.c.term_id, q1.c.val, q1.c.grade, q1.c.number_of_credit,
                q1.c.psubject_code, q1.c.fz_term, fu_user.id)\
        .join_from(q1, fu_user, q1.c.student_login == fu_user.user_login)\
        .subquery()

    q = select(q2.c.student_login, fz_list_student.student_code, q2.c.term_id,
               fz_list_student.major_id,
               fz_list_student.campus_code, q2.c.val, 
               func.sum(q2.c.grade * q2.c.number_of_credit) / func.sum(q2.c.number_of_credit),
               func.substr(q2.c.psubject_code, 1, 3))\
        .join_from(fz_list_student, q2, and_(q2.c.fz_term == fz_list_student.term_id,
                                             q2.c.user_code == fz_list_student.student_code))\
        .where(
               fz_list_student.semester < 3, 
               fz_list_student.student_code.in_(
                   select(fz_list_student.student_code)\
                    .where(fz_list_student.major_id.in_([1, 3, 6, 12, 13, 14, 15, 16, 17,
                                                        28, 29, 30, 31, 32, 33, 34, 35, 
                                                        50, 51, 52, 53, 54, 71, 72]),
                            fz_list_student.campus_code == 'ph',
                            fz_list_student.semester.in_([0, 1, 2]))\
                    .group_by(fz_list_student.student_code)\
                    .having(func.max(fz_list_student.semester) == 2, func.count(distinct(fz_list_student.semester)) >= 2)
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(fz_list_student.campus_code != 'ph')
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(and_(fz_list_student.semester.in_([0, 1, 2]),
                                                                   fz_list_student.status == 'THO'))
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(or_(fz_list_student.term_id < 26,
                                                                  fz_list_student.term_id == 54))
               ))\
        .group_by(fz_list_student.student_code, func.substr(q2.c.psubject_code, 1, 3))
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def check_review(engine, partitioner, mod_number):
    q1 = select(t7_course_result.groupid, t7_course_result.student_login,
                t7_course_result.term_id, t7_course_result.val,
                t7_course_result.grade, t7_course_result.number_of_credit,
                t7_course_result.psubject_code,
                case((~mapping_term.ph.is_(None), mapping_term.ho_term_id), else_=t7_course_result.term_id).label('fz_term'))\
        .join(mapping_term, t7_course_result.term_id == mapping_term.ph, isouter=True)\
        .where(t7_course_result.term_id >= 24)\
        .subquery()
    
    q2 = select(q1.c.groupid, q1.c.student_login, fu_user.user_code,
                q1.c.term_id, q1.c.val, q1.c.grade, q1.c.number_of_credit,
                q1.c.psubject_code, q1.c.fz_term, fu_user.id)\
        .join_from(q1, fu_user, q1.c.student_login == fu_user.user_login)\
        .subquery()
    
    q = select(q2.c.student_login, fz_list_student.student_code, fz_list_student.semester,
               q2.c.term_id, fz_list_student.major_id,
               fz_list_student.campus_code, q2.c.val, 
               fu_group_member.groupid, fu_group.psubject_code, fu_group_member.loai)\
        .join_from(fz_list_student, q2, and_(q2.c.fz_term == fz_list_student.term_id,
                                             q2.c.user_code == fz_list_student.student_code))\
        .join_from(q2, fu_group_member, and_(q2.c.student_login == fu_group_member.member_login,
                                             q2.c.groupid == fu_group_member.groupid))\
        .join_from(fu_group_member, fu_group, fu_group_member.groupid == fu_group.id)\
        .where(
               fz_list_student.semester < 3, 
               fz_list_student.student_code.in_(
                   select(fz_list_student.student_code)\
                    .where(fz_list_student.major_id.in_([1, 3, 6, 12, 13, 14, 15, 16, 17,
                                                        28, 29, 30, 31, 32, 33, 34, 35, 
                                                        50, 51, 52, 53, 54, 71, 72]),
                            fz_list_student.campus_code == 'ph',
                            fz_list_student.semester.in_([0, 1, 2]))\
                    .group_by(fz_list_student.student_code)\
                    .having(func.max(fz_list_student.semester) == 2, func.count(distinct(fz_list_student.semester)) >= 2)
               ),
                              ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(and_(fz_list_student.semester.in_([0, 1, 2]),
                                                                   fz_list_student.status == 'THO'))
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(fz_list_student.campus_code != 'ph')
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(or_(fz_list_student.term_id < 26,
                                                                  fz_list_student.term_id == 54))
               ))
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def fail_count(engine, partitioner, mod_number, semester_range):
    q1 = select(t7_course_result.groupid, t7_course_result.student_login,
                t7_course_result.term_id, t7_course_result.val,
                t7_course_result.grade, t7_course_result.number_of_credit,
                t7_course_result.psubject_code,
                case((~mapping_term.ph.is_(None), mapping_term.ho_term_id), else_=t7_course_result.term_id).label('fz_term'))\
        .join(mapping_term, t7_course_result.term_id == mapping_term.ph, isouter=True)\
        .where(t7_course_result.term_id >= 24)\
        .subquery()
    
    q2 = select(q1.c.groupid, q1.c.student_login, fu_user.user_code,
                q1.c.term_id, q1.c.val, q1.c.grade, q1.c.number_of_credit,
                q1.c.psubject_code, q1.c.fz_term, fu_user.id)\
        .join_from(q1, fu_user, q1.c.student_login == fu_user.user_login)\
        .subquery()

    q = select(q2.c.student_login, fz_list_student.student_code, 
               func.max(fz_list_student.semester), q2.c.term_id, fz_list_student.major_id,
               fz_list_student.campus_code, func.count(q2.c.val))\
        .join_from(fz_list_student, q2, and_(q2.c.fz_term == fz_list_student.term_id,
                                             q2.c.user_code == fz_list_student.student_code))\
        .where(
               fz_list_student.semester < 3, 
               fz_list_student.student_code.in_(
                   select(fz_list_student.student_code)\
                    .where(fz_list_student.major_id.in_([1, 3, 6, 12, 13, 14, 15, 16, 17,
                                                        28, 29, 30, 31, 32, 33, 34, 35, 
                                                        50, 51, 52, 53, 54, 71, 72]),
                            fz_list_student.campus_code == 'ph',
                            fz_list_student.semester.in_([0, 1, 2]))\
                    .group_by(fz_list_student.student_code)\
                    .having(func.max(fz_list_student.semester) == 2, func.count(distinct(fz_list_student.semester)) >= 2)
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(fz_list_student.campus_code != 'ph')
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(and_(fz_list_student.semester.in_([0, 1, 2]),
                                                                   fz_list_student.status == 'THO'))
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(or_(fz_list_student.term_id < 26,
                                                                  fz_list_student.term_id == 54))
               ), q2.c.val != 1)\
        .group_by(fz_list_student.student_code)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def fail_count_by_prefix(engine, partitioner, mod_number, semester_range):
    q1 = select(t7_course_result.groupid, t7_course_result.student_login,
                t7_course_result.term_id, t7_course_result.val,
                t7_course_result.grade, t7_course_result.number_of_credit,
                t7_course_result.psubject_code,
                case((~mapping_term.ph.is_(None), mapping_term.ho_term_id), else_=t7_course_result.term_id).label('fz_term'))\
        .join(mapping_term, t7_course_result.term_id == mapping_term.ph, isouter=True)\
        .where(t7_course_result.term_id >= 24)\
        .subquery()
    
    q2 = select(q1.c.groupid, q1.c.student_login, fu_user.user_code,
                q1.c.term_id, q1.c.val, q1.c.grade, q1.c.number_of_credit,
                q1.c.psubject_code, q1.c.fz_term, fu_user.id)\
        .join_from(q1, fu_user, q1.c.student_login == fu_user.user_login)\
        .subquery()

    q = select(q2.c.student_login, fz_list_student.student_code, 
               func.max(fz_list_student.semester), fz_list_student.major_id,
               fz_list_student.campus_code, func.count(q2.c.val),
               func.substr(q2.c.psubject_code, 1, 3))\
        .join_from(fz_list_student, q2, and_(q2.c.fz_term == fz_list_student.term_id,
                                             q2.c.user_code == fz_list_student.student_code))\
        .where(
               fz_list_student.semester < 3, 
               fz_list_student.student_code.in_(
                   select(fz_list_student.student_code)\
                    .where(fz_list_student.major_id.in_([1, 3, 6, 12, 13, 14, 15, 16, 17,
                                                        28, 29, 30, 31, 32, 33, 34, 35, 
                                                        50, 51, 52, 53, 54, 71, 72]),
                            fz_list_student.campus_code == 'ph',
                            fz_list_student.semester.in_([0, 1, 2]))\
                    .group_by(fz_list_student.student_code)\
                    .having(func.max(fz_list_student.semester) == 2, func.count(distinct(fz_list_student.semester)) >= 2)
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(fz_list_student.campus_code != 'ph')
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(and_(fz_list_student.semester.in_([0, 1, 2]),
                                                                   fz_list_student.status == 'THO'))
               ),
               ~fz_list_student.student_code.in_(
                   select(fz_list_student.student_code).where(or_(fz_list_student.term_id < 26,
                                                                  fz_list_student.term_id == 54))
               ), q2.c.val != 1)\
        .group_by(fz_list_student.student_code, func.substr(q2.c.psubject_code, 1, 3))
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res

async def subject_list(engine, partitioner, mod_number):
    q = select(fu_subject.id, fu_subject.subject_name, 
               fu_subject.subject_code, fu_subject.skill_code,
               fu_subject.num_of_credit)\
        .where(fu_subject.department_id == 28)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res

async def main():
    PARTITIONER = 19

    engine = create_async_engine('mysql+aiomysql://root@localhost/db_test', pool_size=10, max_overflow=20)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    total_mod_numbers = list(range(PARTITIONER))

    # for index, mod_numbers in enumerate(batched(total_mod_numbers, n=20)):
    #     tasks1 = [total_activity(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
    #     res1 = await asyncio.gather(*tasks1)

    #     tasks2 = [total_attendance(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
    #     res2 = await asyncio.gather(*tasks2)

        # tasks3 = [check_user_semester(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
        # res3 = await asyncio.gather(*tasks3)

    #     rows1 = []
    #     rows2 = []
        # rows3 = []

    #     for i in res1:
    #         for j in i:
    #             rows1.append(dict(term_id=j[0], member_login=j[1], member_code=j[2],
    #                         total_act=j[3], term_name=j[4], group_id=j[5]))
    #     for i in res2:
    #         for j in i:
    #             rows2.append(dict(attendance_total=j[0], user_login=j[1], user_code=j[2],
    #                         term_id=j[3], term_name=j[4]))
        # for i in res3:
        #     for j in i:
        #         rows3.append(dict(user_login=j[0], user_code=j[1], semester=j[2],
        #                     term_id=j[3], group_id=j[4]))
                
        # async with engine.connect() as conn:
    #         await conn.execute(insert(sub1), rows1)
    #         await conn.execute(insert(sub2), rows2)
            # await conn.execute(insert(sub3), rows3)
            # await conn.commit()

    # for index, mod_numbers in enumerate(batched(total_mod_numbers, n=20)):
    #     tasks3 = [check_user_semester(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
    #     res3 = await asyncio.gather(*tasks3)
    #     return res3
    
    # for index, mod_numbers in enumerate(batched(total_mod_numbers, n=1)):
    #     tasks = [percent_attendance(engine, PARTITIONER, mod_number, [0, 1, 2, 3]) for mod_number in mod_numbers]
    #     res = await asyncio.gather(*tasks)
    #     return res
    
    # for index, mod_numbers in enumerate(batched(total_mod_numbers, n=1)):
    #     tasks = [avg_grade(engine, PARTITIONER, mod_number, [1, 2]) for mod_number in mod_numbers]
    #     res = await asyncio.gather(*tasks)
    #     return res
    
    # for index, mod_numbers in enumerate(batched(total_mod_numbers, n=1)):
    #     tasks = [avg_grade_by_subject(engine, PARTITIONER, mod_number, [0, 1, 2, 3]) for mod_number in mod_numbers]
    #     res = await asyncio.gather(*tasks)
    #     return res
    
    # for index, mod_numbers in enumerate(batched(total_mod_numbers, n=1)):
    #     tasks = [check_status(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
    #     res = await asyncio.gather(*tasks)
    #     return res

    # for index, mod_numbers in enumerate(batched(total_mod_numbers, n=1)):
    #     tasks = [subject_list(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
    #     res = await asyncio.gather(*tasks)
    #     return res

    # for index, mod_numbers in enumerate(batched(total_mod_numbers, n=1)):
    #     tasks = [check_review(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
    #     res = await asyncio.gather(*tasks)
    #     return res

    # for index, mod_numbers in enumerate(batched(total_mod_numbers, n=1)):
    #     tasks = [fail_count(engine, PARTITIONER, mod_number, [0, 1, 2, 3]) for mod_number in mod_numbers]
    #     res = await asyncio.gather(*tasks)
    #     return res
    
    for index, mod_numbers in enumerate(batched(total_mod_numbers, n=1)):
        tasks = [fail_count_by_prefix(engine, PARTITIONER, mod_number, [0, 1, 2, 3]) for mod_number in mod_numbers]
        res = await asyncio.gather(*tasks)
        return res

    # Clean async engine       
    await engine.dispose()
    
if __name__ == '__main__':
    start_time = time()
    res = asyncio.run(main())
    with open('test.csv', 'w', newline='', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["Student_Code", "Status", "Semester",
    #                          "Major_ID", "Campus_Code", "Term_ID"])
        # csv_writer.writerow(["Term", "User_Login", "User_Code", "Total_Attendance", "Total_Activity", "Percentage", "Up_To_Semester"])
        # csv_writer.writerow(["User_Login", "User_Code", "Up_To_Semester", "Term_ID", "Major_ID", "Campus",
        #                      "Value", "Total_Grade", "Total_Credit", "Average_Grade"])
        # csv_writer.writerow(["Term_ID", "User_Login", 
        # "Fail_ACC", "Avg_ACC", "Fail_AND", "Avg_AND", "Fail_ASI", "Avg_ASI", "Fail_AUT", "Avg_AUT", 
        # "Fail_BUS", "Avg_BUS", "Fail_CDI", "Avg_CDI", "Fail_CHE", "Avg_CHE", "Fail_COM", "Avg_COM", 
        # "Fail_CRO", "Avg_CRO", "Fail_DAT", "Avg_DAT", "Fail_DOM", "Avg_DOM", "Fail_EHO", "Avg_EHO", 
        # "Fail_ENG", "Avg_ENG", "Fail_ENT", "Avg_ENT", "Fail_ETO", "Avg_ETO", "Fail_FIN", "Avg_FIN", 
        # "Fail_GAM", "Avg_GAM", "Fail_GRE", "Avg_GRE", "Fail_HIS", "Avg_HIS", "Fail_HOS", "Avg_HOS", 
        # "Fail_HUR", "Avg_HUR", "Fail_INE", "Avg_INE", "Fail_INF", "Avg_INF", "Fail_IOT", "Avg_IOT", 
        # "Fail_ITI", "Avg_ITI", "Fail_KBA", "Avg_KBA", "Fail_KBS", "Avg_KBS", "Fail_KOT", "Avg_KOT", 
        # "Fail_LOG", "Avg_LOG", "Fail_MAN", "Avg_MAN", "Fail_MAR", "Avg_MAR", "Fail_MAT", "Avg_MAT", 
        # "Fail_MBS", "Avg_MBS", "Fail_MEC", "Avg_MEC", "Fail_MOB", "Avg_MOB", "Fail_MUL", "Avg_MUL", 
        # "Fail_NET", "Avg_NET", "Fail_PDP", "Avg_PDP", "Fail_PHY", "Avg_PHY", "Fail_PMA", "Avg_PMA", 
        # "Fail_PR0", "Avg_PR0", "Fail_PRE", "Avg_PRE", "Fail_PRO", "Avg_PRO", "Fail_PSY", "Avg_PSY", 
        # "Fail_SKI", "Avg_SKI", "Fail_SOA", "Avg_SOA", "Fail_SOF", "Avg_SOF", "Fail_SUP", "Avg_SUP", 
        # "Fail_SYB", "Avg_SYB", "Fail_TES", "Avg_TES", "Fail_TOU", "Avg_TOU", "Fail_VIE", "Avg_VIE", 
        # "Fail_WEB", "Avg_WEB","Up_To_Semester", "Total_Semester"])
        # csv_writer.writerow(["Member_Login", "User_Code", "Value", "Subject_Code", "Skill_Code",
        #                      "Semester", "Number_Of_Credit", "Average_Grade", "Number_Of_Attempt"])
        # csv_writer.writerow(["ID", "Subject_Name", "Subject_Code", "Skill_Code", "Number_Of_Credit"])
        # csv_writer.writerow(["Student_Code", "Status", "Semester", "Term_ID", "Major_ID", "Campus_Code",
        #                      "Value", "Average_Grade", "Subject_Code"])
        # csv_writer.writerow(["Student_Code", "Status", "Up_To_Semester", "Campus_Code",
        #                      "Value", "Total_Grade", "Total_Credit", "Average_Grade"])
        # csv_writer.writerow(["User_Login", "User_Code", "Term_ID", "Major_ID", "Campus",
        #                      "Value", "Average_Grade", "Prefix_Subject"])
        # csv_writer.writerow(["User_Login", "User_Code", "Semester", "Term_ID",
        #                      "Majod_ID", "Campus", "Value", "Group_ID",
        #                      "Subject_Code", "Type"])
        # csv_writer.writerow(["User_Login", "User_Code", "Up_To_Semester", "Term_ID",
        #                      "Major_ID", "Campus", "Total_Fail"])
        csv_writer.writerow(["User_Login", "User_Code", "Term_ID",
                             "Major_ID", "Campus", "Total_Fail", "Subject"])
        process_data(res, csv_writer)
    
    """
    drop subject that has all null value of less than 5% total

    df = pd.read_csv('test.csv')
    avg_null_col = df.filter(like='Avg').isnull().all()
    fail_null_col = [col.replace('Avg', 'Fail') for col, is_null in avg_null_col.items() if is_null]
    drop_col = list(avg_null_col[avg_null_col].index) + fail_null_col
    df = df.drop(columns=drop_col, axis=1)

    threshold = len(df)*0.01
    avg_drop_col = []
    for col in df.columns:
        if 'Avg' in col and df[col].notnull().sum() < threshold:
            avg_drop_col.append(col)
    fail_drop_col = [col.replace('Avg', 'Fail') for col in avg_drop_col]
    drop_col = avg_drop_col + fail_drop_col
    df = df.drop(columns=drop_col, axis=1)

    df.to_csv('pandas.csv', sep=',', index=False, encoding='utf-8')
    """

    # df = pd.read_csv('test.csv')
    # df["Type"] = df["Type"].fillna(3).astype(int)
    # print(df.dtypes)
    # df.to_csv('after.csv', sep=',', index=False, encoding='utf-8')

    # df = pd.read_csv('fail_count_by_prefix.csv')
    # count_fail = df.groupby("Subject").size()
    # print(count_fail)

    print(f"Time taken: {time() - start_time}")