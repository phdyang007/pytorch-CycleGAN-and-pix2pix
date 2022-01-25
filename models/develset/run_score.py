 #!/home/guojin/miniconda3/envs/hdevel/bin/python
import os
import numpy as np
# import train
if __name__ == "__main__":
    py = '/home/guojin/miniconda3/envs/hdevel/bin/python'
    for ilt in [0.5, 1, 1.5]:
        for pvb in [0.1, 0.5, 1, 3, 5, 7]:
            for curv in [True, False]:
                for exp in [2, 4]:
                    print(f'{py} train.py \
                        model.loss_weights.ilt_weight={ilt} \
                        model.loss_weights.pvb_weight={pvb} \
                        model.ilt_exp={exp} model.add_curv={str(curv)}')
                    # os.system(f'conda activate hdevel && python train.py model.loss_weights.ilt_weight={ilt} model.loss_weights.pvb_weight={pvb} model.ilt_exp={exp} model.add_curv={str(curv)}')
