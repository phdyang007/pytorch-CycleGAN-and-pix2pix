import sqlite3
class RESULT_DB:
    def __init__(self, db_path):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        # self.cursor.execute("DROP TABLE IF EXISTS result")
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS result
        (case_name TEXT,
        ilt_weight REAL,
        pvb_weight REAL,
        add_curv TEXT,
        curv_weight REAL,
        ilt_exp INTEGER,
        epoch INTEGER,
        L2 REAL,
        pv_band REAL,
        runtime REAL,
        epe INTEGER,
        min_shots INTEGER)''')
        print(f'new database {db_path} created.')

    def insert_record(
        self,
        case_name,
        ilt_weight,
        pvb_weight,
        add_curv,
        curv_weight,
        ilt_exp,
        epoch,
        L2,
        pv_band,
        runtime,
        epe = -100,
        min_shots = -100
        ):
        self.cursor.execute("INSERT INTO result VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')".
                            format(case_name,
                            ilt_weight,
                            pvb_weight,
                            add_curv,
                            curv_weight,
                            ilt_exp,
                            epoch,
                            L2,
                            pv_band,
                            runtime,
                            epe,
                            min_shots))
        self.connection.commit()

    def close(self):
        self.connection.close()


class CPP_RESULT_DB:
    def __init__(self, db_path):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        # self.cursor.execute("DROP TABLE IF EXISTS cppresult")
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS cppresult
        (case_name TEXT,
        ilt_weight REAL,
        pvb_weight REAL,
        vel_weight REAL,
        add_curv TEXT,
        curv_weight REAL,
        ilt_exp INTEGER,
        epoch INTEGER,
        L2 REAL,
        pv_band REAL,
        epe INTEGER,
        img_path TEXT UNIQUE,
        s_score REAL,
        c_score REAL,
        min_shots INTEGER)''')
        print(f'new database {db_path} created.')

    def insert_record(
        self,
        case_name,
        ilt_weight,
        pvb_weight,
        vel_weight,
        add_curv,
        curv_weight,
        ilt_exp,
        epoch,
        L2,
        pv_band,
        epe,
        img_path,
        s_score,
        c_score,
        min_shots = -100
        ):
        self.cursor.execute("INSERT OR IGNORE INTO cppresult VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}','{}', '{}', '{}', '{}')".
                            format(case_name,
                            ilt_weight,
                            pvb_weight,
                            vel_weight,
                            add_curv,
                            curv_weight,
                            ilt_exp,
                            epoch,
                            L2,
                            pv_band,
                            epe,
                            img_path,
                            s_score,
                            c_score,
                            min_shots))
        self.connection.commit()

    def close(self):
        self.connection.close()