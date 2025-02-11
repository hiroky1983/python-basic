# 数学学習カテゴリ
MATH_CATEGORIES = {
    '基礎数学': [
        '整数と小数',
        '分数と比',
        '方程式と不等式',
        '図形の基礎',
        '確率と統計の基礎'
    ],
    
    '代数学': [
        '多項式',
        '二次関数',
        '指数と対数',
        '複素数',
        '行列'
    ],
    
    '幾何学': [
        '平面図形',
        '空間図形',
        '三角比',
        'ベクトル',
        '座標幾何'
    ],
    
    '解析学': [
        '数列と級数',
        '微分',
        '積分',
        '微分方程式',
        'フーリエ解析'
    ],
    
    '応用数学': [
        '統計学',
        '線形代数',
        '数値解析',
        '最適化理論',
        '暗号理論'
    ]
}

# カテゴリの取得
def get_math_categories():
    return list(MATH_CATEGORIES.keys())

# サブカテゴリの取得
def get_subcategories(category):
    return MATH_CATEGORIES.get(category, [])
