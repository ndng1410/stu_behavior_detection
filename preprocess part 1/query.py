from sqlalchemy import create_engine, func, Column, Integer, String, ForeignKey, DATETIME, TIMESTAMP, BigInteger, SmallInteger, VARCHAR
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.pool import QueuePool
from time import time
import multiprocessing
import csv
import numpy as np

Base = declarative_base()

engine = create_engine('mysql+mysqlconnector://root@localhost/db_test', echo=False, poolclass=QueuePool)

Base.metadata.create_all(engine)

class fu_group(Base):
    __tablename__= 'fu_group'
    id = Column(BigInteger, primary_key=True)
    pterm_id = Column(Integer)
    pterm_name = Column(VARCHAR)
    is_virtual = Column(SmallInteger)
    psubject_name = Column(VARCHAR)

class fu_group_member(Base):
    __tablename__ = 'fu_group_member'
    id = Column(BigInteger, primary_key=True)
    groupid = Column(Integer)
    member_login = Column(VARCHAR)

class fu_activity(Base):
    __tablename__ = 'fu_activity'
    id = Column(BigInteger, primary_key=True)
    groupid = Column(Integer)
    course_slot = Column(SmallInteger)
    term_id = Column(Integer)

class fu_attendace(Base):
    __tablename__ = 'fu_attendance'
    id = Column(BigInteger, primary_key=True)
    activity_id = Column(BigInteger)
    user_login = Column(VARCHAR)
    groupid = Column(Integer)

class fu_user(Base):
    __tablename__ = 'fu_user'
    id = Column(BigInteger, primary_key=True)
    user_login = Column(VARCHAR)
    user_code = Column(VARCHAR)

class fz_list_student(Base):
    __tablename__ = 'fz_list_student'
    id = Column(BigInteger, primary_key=True)
    student_code = Column(VARCHAR)
    semester = Column(Integer)
    term_id = Column(Integer)

def group_member(page_size, page_number, result_queue):
    try:
        offset = page_size * (page_number - 1)
        with Session(engine) as session:
            result = session.query(fu_group.id, fu_group.pterm_id, fu_group_member.member_login)\
                .join(fu_group_member, fu_group_member.groupid == fu_group.id)\
                .filter(fu_group.pterm_id >= 24, fu_group.is_virtual == 0)\
                .offset(offset).limit(page_size).all()
            return result
    except Exception as e:
        print(f'Error {e}')
        return []

def group_activity(page_size, page_number, result_queue):
    try:
        offset = page_size * (page_number - 1)
        with Session(engine) as session:
            result = session.query(fu_activity.groupid, fu_activity.term_id, func.count(fu_activity.id))\
                .join(fu_group, fu_group.id == fu_activity.groupid)\
                .filter(fu_group.pterm_id >= 24, fu_group.is_virtual == 0, fu_activity.course_slot < 100)\
                .group_by(fu_activity.groupid)\
                .offset(offset).limit(page_size).all()
            return result
    except Exception as e:
        print(f'Error {e}')
        return []
    
def check_act(group_member, group_activity):
    try:
        group_member_extr = [item for sublist in group_member for item in sublist]
        group_activity_extr = [item for sublist in group_activity for item in sublist]
        join_result = [(m[0], m[1], m[2], a[2]) for m in group_member_extr for a in group_activity_extr if m[0]==a[0]]
        return join_result
    except Exception as e:
        print(f'Error {e}')
        return []

def check_attendance(page_size, page_number, result_queue):
    try:
        offset = page_size * (page_number - 1)
        with Session(engine) as session:
            result = session.query(func.count(fu_attendace.activity_id), fu_attendace.user_login, fu_group.pterm_id)\
            .join(fu_group, fu_attendace.groupid == fu_group.id)\
            .filter(fu_group.pterm_id >= 24, fu_group.is_virtual == 0)\
            .group_by(fu_group.pterm_id, fu_attendace.user_login)\
            .offset(offset).limit(page_size).all()
        # result_queue.put(result)
        return result
    except Exception as e:
        print(f'Error {e}')
        return []

def user_code(page_size, page_number, result_queue):
    try:
        offset = page_size * (page_number - 1)
        with Session(engine) as session:
            result = session.query(fu_user.user_login, fu_user.user_code, fz_list_student.semester,\
                fz_list_student.term_id)\
                .join(fz_list_student, fz_list_student.student_code == fu_user.user_code)\
                .filter(fz_list_student.term_id >= 24)\
                .order_by(fz_list_student.term_id, fz_list_student.semester)\
                .offset(offset).limit(page_size).all()
        # result_queue.put(result)
        return result
    except Exception as e:
        print(f'Error {e}')
        return []

def process_data(chunk, csv_writer):
    for i in chunk:
        csv_writer.writerow(i)

if __name__ == '__main__':
    page_size = 10
    total_pages = 1

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    with multiprocessing.Pool(processes=8) as pool:
        start_time = time()
        page_args = [(page_size, page_number, result_queue) for page_number in range(1, total_pages + 1)]
        res_group_member = pool.starmap(check_attendance, page_args)
        pool.close()
        pool.join()

    for i in res_group_member:
        print(i)  

    with open('test2.csv', 'w', newline='', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        while not result_queue.empty():
            result_chunk = result_queue.get()
            process_data(result_chunk, csv_writer)

    print(f"Time taken: {time() - start_time}")

