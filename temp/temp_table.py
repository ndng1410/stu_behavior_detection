from sqlalchemy import text, select, and_, or_, insert, func, Column, Integer, BigInteger, SmallInteger, VARCHAR
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine
from time import time
import csv
import asyncio
import nest_asyncio
from itertools import islice

nest_asyncio.apply()

Base = declarative_base()

class fu_group(Base):
    __tablename__= 'fu_group'
    id = Column(BigInteger, primary_key=True)
    pterm_id = Column(Integer, index=True)
    pterm_name = Column(VARCHAR, index=True)
    is_virtual = Column(SmallInteger, index=True)
    psubject_name = Column(VARCHAR, index=True)

class fu_group_member(Base):
    __tablename__ = 'fu_group_member'
    id = Column(BigInteger, primary_key=True)
    groupid = Column(Integer, index=True)
    member_login = Column(VARCHAR, index=True)

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

class sub1(Base):
    __tablename__ = 'total_activity'
    __table_args__ = {'prefixes': ['TEMPORARY']}
    id = Column(BigInteger, primary_key=True)
    fu_group_termid = Column(Integer, nullable=False, index=True)
    fu_group_term_name = Column(VARCHAR(100), nullable=False, index=True)
    fu_group_member_login = Column(VARCHAR(100), nullable=False, index=True)
    fu_group_total_act = Column(Integer, nullable=False, index=True)

class sub2(Base):
    __tablename__ = 'total_attendance'
    __table_args__ = {'prefixes': ['TEMPORARY']}
    id = Column(BigInteger, primary_key=True)
    fu_group_term_name = Column(VARCHAR(100), nullable=False, index=True)
    fu_attendance_total = Column(Integer, nullable=False, index=True)
    fu_attendance_user = Column(VARCHAR(100), nullable=False, index=True)
    fu_attendance_termid = Column(Integer, nullable=False, index=True)

class sub3(Base):
    __tablename__ = 'total_user_semester'
    __table_args__ = {'prefixes': ['TEMPORARY']}
    id = Column(BigInteger, primary_key=True)
    fz_user_login = Column(VARCHAR(100), nullable=False, index=True)
    fz_semester = Column(Integer, nullable=False, index=True)
    fz_termid = Column(Integer, nullable=False, index=True)

async def total_activity(engine, partitioner, mod_number):
    q1 = select(fu_group.id, fu_group.pterm_id, fu_group_member.member_login, fu_group.pterm_name)\
        .join_from(fu_group_member, fu_group, fu_group.id == fu_group_member.groupid)\
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
    q = select(q1.c.pterm_id, q1.c.member_login, func.sum(q2.c.act_total).label('total_act'), q1.c.pterm_name)\
        .join_from(q1, q2, q1.c.id == q2.c.groupid)\
        .where(q1.c.pterm_id % 11 == mod_number)\
        .group_by(q1.c.pterm_id, q1.c.member_login)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def total_attendance(engine, partitioner, mod_number):
    q1 = select(fu_group.id, fu_group.pterm_id, fu_group.pterm_name)\
        .where(fu_group.pterm_id >= 24,
               fu_group.is_virtual == 0
               ).subquery()
    q = select(func.count(fu_attendance.activity_id).label('total_att'), fu_attendance.user_login, q1.c.pterm_id, q1.c.pterm_name)\
        .join_from(fu_attendance, q1, fu_attendance.groupid == q1.c.id)\
        .where(fu_attendance.val == 1,
               q1.c.pterm_id % 17 == mod_number)\
        .group_by(fu_attendance.user_login, q1.c.pterm_id)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def check_user_semester(engine, partitioner, mod_number):
    q = select(fu_user.user_login, fz_list_student.semester, fz_list_student.term_id)\
        .join_from(fz_list_student, fu_user, fz_list_student.student_code == fu_user.user_code)\
        .where(fz_list_student.term_id >= 24,
               fu_user.id % 13 == mod_number)
    async with engine.connect() as conn:
        result = await conn.execute(q)
        res = result.all()
        return res
    
async def percent_attendance(engine, partitioner, mod_number, semester_range):
    q = select(sub1.fu_group_termid, sub1.fu_group_member_login,
               func.sum(sub2.fu_attendance_total), func.sum(sub1.fu_group_total_act),
               func.sum(sub2.fu_attendance_total) / func.sum(sub1.fu_group_total_act),
               func.max(sub3.fz_semester), sub1.fu_group_term_name)\
        .join_from(sub1, sub2, and_(sub1.fu_group_termid == sub2.fu_attendance_termid,
                                    sub1.fu_group_member_login == sub2.fu_attendance_user))\
        .join_from(sub1, sub3, and_(sub1.fu_group_termid == sub3.fz_termid,
                                    sub1.fu_group_member_login == sub3.fz_user_login))\
        .where(and_(sub1.fu_group_termid % 8 == mod_number, sub3.fz_semester.in_(semester_range)))\
        .group_by(sub1.fu_group_member_login)\
        .having(or_(func.count(sub3.fz_semester) > 1, func.min(sub3.fz_semester) > 1))
    result = await engine.execute(q)
    res = result.all()
    return res

"""
query condition:
- select with only semester ?
- select with both semester and specific term (ex: Spring/Summer/Fall 20xx) ?
- duplicate semester with different termid ?
- duplicate code for different scenerio ?
"""
    
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

async def create_temporary_tables(engine):
    async with engine.connect() as conn:
        q1 = """
            CREATE TEMPORARY TABLE IF NOT EXISTS total_activity (
                id INTEGER AUTO_INCREMENT NOT NULL,
                fu_group_term_name VARCHAR(100) NOT NULL,
                fu_group_termid INTEGER NOT NULL,
                fu_group_member_login VARCHAR(100) NOT NULL,
                fu_group_total_act INTEGER NOT NULL,
                PRIMARY KEY (id),
                INDEX (fu_group_termid, fu_group_member_login, fu_group_total_act)
            )
        """
        await conn.execute(text(q1))

        q2 = """
            CREATE TEMPORARY TABLE IF NOT EXISTS total_attendance (
                id INTEGER AUTO_INCREMENT NOT NULL,
                fu_group_term_name VARCHAR(100) NOT NULL,
                fu_attendance_total INTEGER NOT NULL,
                fu_attendance_user VARCHAR(100) NOT NULL,
                fu_attendance_termid INTEGER NOT NULL,
                PRIMARY KEY (id),
                INDEX (fu_attendance_total, fu_attendance_user, fu_attendance_termid)
            )
        """
        await conn.execute(text(q2))

        q3 = """
            CREATE TEMPORARY TABLE IF NOT EXISTS total_user_semester (
                id INTEGER AUTO_INCREMENT NOT NULL,
                fz_user_login VARCHAR(100) NOT NULL,
                fz_semester INTEGER NOT NULL,
                fz_termid INTEGER NOT NULL,
                PRIMARY KEY (id),
                INDEX (fz_user_login, fz_semester, fz_termid)
            )
        """
        await conn.execute(text(q3))

async def main():
    PARTITIONER = 19

    engine = create_async_engine('mysql+aiomysql://root@localhost/db_test', pool_size=10, max_overflow=20)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    total_mod_numbers = list(range(PARTITIONER))
    # Split up jobs so we don't overload the connection pool.
    await create_temporary_tables(engine)

    async with engine.connect() as conn:
        for index, mod_numbers in enumerate(batched(total_mod_numbers, n=20)):
            tasks1 = [total_activity(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
            res1 = await asyncio.gather(*tasks1)

            tasks2 = [total_attendance(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
            res2 = await asyncio.gather(*tasks2)

            tasks3 = [check_user_semester(engine, PARTITIONER, mod_number) for mod_number in mod_numbers]
            res3 = await asyncio.gather(*tasks3)

            rows1 = []
            rows2 = []
            rows3 = []

            for i in res1:
                for j in i:
                    rows1.append(dict(fu_group_termid=j[0], fu_group_member_login=j[1],
                                fu_group_total_act=j[2], fu_group_term_name=j[3]))
            for i in res2:
                for j in i:
                    rows2.append(dict(fu_attendance_total=j[0], fu_attendance_user=j[1],
                                fu_attendance_termid=j[2], fu_group_term_name=j[3]))
            for i in res3:
                for j in i:
                    rows3.append(dict(fz_user_login=j[0], fz_semester=j[1],
                                fz_termid=j[2]))
                    
            await conn.execute(insert(sub1), rows1)
            await conn.execute(insert(sub2), rows2)
            await conn.execute(insert(sub3), rows3)
            await conn.commit()

        for index, mod_numbers in enumerate(batched(total_mod_numbers, n=20)):
            tasks = [percent_attendance(conn, PARTITIONER, mod_number, [1, 2]) for mod_number in mod_numbers]
            res = await asyncio.gather(*tasks)
            return res

    # Clean async engine       
    await engine.dispose()
    
if __name__ == '__main__':
    start_time = time()
    res = asyncio.run(main())
    with open('test.csv', 'w', newline='', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Term", "User_Login", "Total_Attendance", "Total_Activity", "Percentage", "Up_To_Semester", "Started_Term"])
        process_data(res, csv_writer)
    print(f"Time taken: {time() - start_time}")