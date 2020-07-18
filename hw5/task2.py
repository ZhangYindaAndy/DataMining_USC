import sys
import time
import random
from binascii import hexlify
from statistics import mean, median
from blackbox import BlackBox

prime = [3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 
         3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 
         3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769,
          3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 
          3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 
          4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 
          4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 
          4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523]

hash_num = 45
hash_group = 5
group_length = int(hash_num / hash_group)
string_length = 2048

def generate_hash_list(hash_num):
    # f(x) = (ax + b) % p % m
    # p is any prime number, m is the number of bins
    random.seed(0)
    a = random.sample(range(1, 10000), hash_num)
    b = random.sample(range(1, 10000), hash_num)
    p = random.sample(prime, hash_num)
    return (a, b, p)

hash_list = generate_hash_list(hash_num)


def myhashs(user):
    # calc_hash = lambda a, x, b: (a * x + b) % 163119
    calc_hash = lambda a, x, b, p: (a * x + b) % p 

    result = []
    user_id = int(hexlify(user.encode('utf-8')), 16)

    for i in range(hash_num):
        hash_val = calc_hash(hash_list[0][i], user_id, hash_list[1][i], hash_list[2][i]) % string_length
        hash_val = bin(hash_val).lstrip('0b')
        result.append(hash_val)
    # print(result)
    
    return result


if __name__ == '__main__':

    start = time.time()

    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]

    bx = BlackBox()
    ground_truth = []
    estimate_length = []

    for _ in range(num_of_asks):
        stream_users = bx.ask(input_path, stream_size)
        ground_truth.append(len(set(stream_users)))
        longest_trail = [0 for _ in range(hash_num)]
        for user in stream_users:
            user_hash = myhashs(user)
            for i, h in enumerate(user_hash):
                longest_trail[i] = max(longest_trail[i], len(h) - len(h.rstrip('0')))
        
        for i in range(hash_num):
            longest_trail[i] = 2 ** longest_trail[i]
        # print(longest_trail)

        avg_trail = []
        for i in range(hash_group):
            avg_trail.append(mean(longest_trail[i*group_length:(i+1)*group_length]))
        # print(avg_trail)
        median_trail = median(avg_trail)
        # print(median_trail)
        estimate_length.append(round(median_trail))

    print(sum(estimate_length))
    print(sum(estimate_length) * 1.0 / sum(ground_truth))


    with open(output_path, 'w') as f:
        f.write('Time,Ground Truth,Estimation\n')
        for i in range(num_of_asks):
            f.write('{},{},{}\n'.format(i, ground_truth[i], estimate_length[i]))

    print('Duration: {}'.format(time.time()-start))



