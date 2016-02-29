import os
import sys

bin_path = sys.argv[1]

count = 0
for length in ['30']:
    sample_size = 2
    for krylov_size in [100]:
        for function in ['0', '1', '2']:
            for dm_file in ['thermalIsing_L30_D20_beta1_0.dat', 'thermalIsing_L30_D20_beta0_1.dat']:
                # for dm_file in ['thermalIsing_L10_D20_beta1_0.dat']:
                # for dm_file in ['thermalIsing_L30_D20_beta0_1.dat']:
                for mpo_bond in [30]:
                    for kry_bond in [10, 30, 50, 80, 120, 150]:
                        # for kry_bond in [200]:
                        print 'EXECUTING ' + bin_path + ' ' + str(krylov_size) + ' ' + str(
                                sample_size) + ' ' + function + ' ' + length + ' ' + dm_file + ' ' + str(
                                mpo_bond) + ' ' + str(kry_bond)
                        with open('./batchfiles/%s_%s_%s_%d_%d_%d_%d.sh' % (
                                length, dm_file, function, krylov_size, sample_size, mpo_bond, kry_bond), 'w') as f:
                            f.write("#!/bin/bash\n\n")
                            f.write("#SBATCH -o /home/hpc/pr63so/gu53dek2/experiments/%s_%s_%s_%d_%d_%d_%d.out\n" % (
                                length, dm_file, function, krylov_size,
                                sample_size, mpo_bond, kry_bond))
                            f.write("#SBATCH -D /home/hpc/pr63so/gu53dek2/experiments\n")
                            f.write("#SBATCH -J %s_%s_%s_%d_%d_%d_%d\n" % (
                                length, dm_file, function, krylov_size, sample_size, mpo_bond, kry_bond))
                            f.writelines(
                                    ["#SBATCH --get-user-env\n", "#SBATCH --partition=snb\n", "#SBATCH --ntasks=1\n",
                                     "#SBATCH --cpus-per-task=16\n",
                                     "#SBATCH --mail-type=END\n", "#SBATCH --mail-user=mumme@in.tum.de\n",
                                     "#SBATCH --export=NONE\n", "#SBATCH --time=24:00:00\n"])
                            f.write("#SBATCH --nodelist=mac-snb[1-10]\n")
                            f.write("\n")
                            f.write(bin_path + ' ' + str(
                                    krylov_size) + ' ' + function + ' ' + length + ' ' + dm_file + ' ' + str(
                                    mpo_bond) + ' ' + str(kry_bond))
                        # os.system("sbatch " + "./batchfiles/%s_%s_%s_%d_%d_%d_%d.sh" % (
                        # length, dm_file, function, krylov_size, sample_size, mpo_bond, kry_bond))
                        count += 1

print 'You should find %d files' % count
