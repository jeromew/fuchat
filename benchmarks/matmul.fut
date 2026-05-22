def naive A B =
  let dot_prod xs ys = f32.sum (map2 (*) xs ys)
  in map (\a -> map (dot_prod a) (transpose B)) A

def smart A B =
  map (\a -> map f32.sum (transpose (map2 (\x -> map (*x)) a B))) A

def gen (n: i64) : [n][n]f32 =
  tabulate n (\i -> tabulate n (\j -> f32.sin (f32.i64 (i + j))))

-- ==
-- entry: test_naive
-- random compiled input { [50][50]f32 [50][50]f32 }
-- auto output
-- random compiled input { [100][100]f32 [100][100]f32 }
-- auto output
-- random compiled input { [250][250]f32 [250][250]f32 }
-- auto output
-- random compiled input { [256][256]f32 [256][256]f32 }
-- auto output
-- random compiled input { [500][500]f32 [500][500]f32 }
-- auto output
-- random compiled input { [512][512]f32 [512][512]f32 }
-- auto output
-- random compiled input { [1000][1000]f32 [1000][1000]f32 }
-- auto output
-- random compiled input { [1024][1024]f32 [1024][1024]f32 }
-- auto output
entry test_naive [n] (A: [n][n]f32) (B: [n][n]f32) : f32 =
  f32.sum (flatten (naive A B))

-- ==
-- entry: test_smart
-- random compiled input { [50][50]f32 [50][50]f32 }
-- auto output
-- random compiled input { [100][100]f32 [100][100]f32 }
-- auto output
-- random compiled input { [250][250]f32 [250][250]f32 }
-- auto output
-- random compiled input { [256][256]f32 [256][256]f32 }
-- auto output
-- random compiled input { [500][500]f32 [500][500]f32 }
-- auto output
-- random compiled input { [512][512]f32 [512][512]f32 }
-- auto output
-- random compiled input { [1000][1000]f32 [1000][1000]f32 }
-- auto output
-- random compiled input { [1024][1024]f32 [1024][1024]f32 }
-- auto output
entry test_smart [n] (A: [n][n]f32) (B: [n][n]f32) : f32 =
  f32.sum (flatten (smart A B))