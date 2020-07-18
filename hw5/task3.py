import sys
import time
import random

from blackbox import BlackBox


if __name__ == '__main__':
    
    start = time.time()

    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]

    bx = BlackBox()
    random.seed(553)
    

    n = 0
    output_user = []
    reservior = []

    for _ in range(num_of_asks):
        stream_users = bx.ask(input_path, stream_size)
        for user in stream_users:
            n += 1
            if len(reservior) < 100:
                reservior.append(user)
            elif random.randint(0, 100000) % n < 100:
                replaced_id = random.randint(0, 100000) % 100
                # print(replaced_id)
                reservior[replaced_id] = user

        # print(n)
        # print([reservior[0], reservior[20], reservior[40], reservior[60], reservior[80]])
        output_user.append([reservior[0], reservior[20], reservior[40], reservior[60], reservior[80]])

    with open(output_path, 'w') as f:
        f.write('seqnum,0_id,20_id,40_id,60_id,80_id\n')
        for i in range(num_of_asks):
            f.write('{},{},{},{},{},{}\n'.format((i+1)*stream_size, output_user[i][0], output_user[i][1], output_user[i][2], output_user[i][3], output_user[i][4]))

    print('Duration: {}'.format(time.time()-start))
