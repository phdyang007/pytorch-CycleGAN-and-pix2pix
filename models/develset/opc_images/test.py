import subprocess
if __name__ == "__main__":
    process = subprocess.Popen(['./mask_fracturing','M1_test1_0.txt'])
    out, err = process.communicate()
    errcode = process.returncode
    print(errcode)
    process.kill()
    process.terminate()