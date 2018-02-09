/*
 * NOTE :
 * 	This program is a JavaScript version of Mersenne Twister,
 * 	conversion from the original program (mt19937ar.c),
 * 	translated by yunos on december, 6, 2008.
 * 	If you have any questions about this program, please ask me by e-mail.
 * 
 * 
 * 
 * Updated 2008/12/08
 * Ver. 1.00
 * charset = UTF8
 * 
 * Mail : info@graviness.com
 * Home : http://www.graviness.com/
 * 
 * 擬似乱数生成器メルセンヌ・ツイスタクラス．
 * 
 * MathクラスのクラスメソッドにmersenneTwisterRandomメソッドを追加します．
 * 
 * Ref.
 * 	http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/mt.html
 */



/**
 * 擬似乱数生成器メルセンヌ・ツイスタクラス．
 * 
 * 擬似乱数生成方法の標準であるメルセンヌ・ツイスタが実装されます．
 * 
 * 符号無し32ビット整数型の一様乱数を基本とし，符号無し46ビット整数型一様乱数，
 * 浮動小数点型の一様乱数を生成します．
 * 乱数生成の初期化には，一つの整数を使用しますが，必要に応じて
 * 配列を用いた任意ビット幅の値を使用することもできます．
 * 
 * このクラスは以下のサイト(C言語ソース)のJavaScript言語移植版です．
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
 * (http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/mt.html)
 * 外部インタフェースは，Javaのjava.util.Randomクラスを参考に実装されています．
 * http://sdc.sun.co.jp/java/docs/j2se/1.4/ja/docs/ja/api/java/util/Random.html
 * 
 * 性能は，ビルトインのMath.randomの約2分の一ですが，
 * 乱数の品質は当該サイトに示す通りです．
 * 
 * 使用例)
 * // インスタンスを生成し，乱数生成器を現在時刻で初期化します．
 * var mt = new MersenneTwister(new Date().getTime());
 * for (var i = 0; i < 1000; ++i) {
 * 	// 32ビット符号無し整数型の一様乱数
 * 	var randomNumber = mt.nextInteger();
 * }
 */
function class__MersenneTwister__(window) {
    var className = "MersenneTwister";

    var $next = "$__next__";

    var N = 624;
    var M = 397;
    var MAG01 = [0x0, 0x9908b0df];

    /**
	 * 新しい乱数ジェネレータを生成します．
	 * 引数に応じたシードを設定します．
	 * 
	 * @param (None)	新しい乱数ジェネレータを生成します．
	 * シードは現在時刻を使用します．
	 * @see Date#getTime()
	 * ---
	 * @param number	
	 * @see #setSeed(number)
	 * ---
	 * @param number[]	
	 * @see #setSeed(number[])
	 * ---
	 * @param number, number, ...	
	 * @see #setSeed(number, number, ...)
	 */
    var F = window[className] = function () {
        this.mt = new Array(N);
        this.mti = N + 1;

        var a = arguments;
        switch (a.length) {
            case 0:
                this.setSeed(new Date().getTime());
                break;
            case 1:
                this.setSeed(a[0]);
                break;
            default:
                var seeds = new Array();
                for (var i = 0; i < a.length; ++i) {
                    seeds.push(a[i]);
                }
                this.setSeed(seeds);
                break;
        }
    };

    var FP = F.prototype;

    /**
	 * 乱数ジェネレータのシードを設定します．
	 * 
	 * @param number	単一の数値を使用し，
	 * 	乱数ジェネレータのシードを設定します．
	 * ---
	 * @param number[]	複数の数値を使用し，
	 * 	乱数ジェネレータのシードを設定します．
	 * ---
	 * @param number, number, ...	複数の数値を使用し，
	 * 	乱数ジェネレータのシードを設定します．
	 */
    FP.setSeed = function () {
        var a = arguments;
        switch (a.length) {
            case 1:
                if (a[0].constructor === Number) {
                    this.mt[0] = a[0];
                    for (var i = 1; i < N; ++i) {
                        var s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
                        this.mt[i] = ((1812433253 * ((s & 0xffff0000) >>> 16))
                                << 16)
                            + 1812433253 * (s & 0x0000ffff)
                            + i;
                    }
                    this.mti = N;
                    return;
                }

                this.setSeed(19650218);

                var l = a[0].length;
                var i = 1;
                var j = 0;

                for (var k = N > l ? N : l; k != 0; --k) {
                    var s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30)
                    this.mt[i] = (this.mt[i]
                            ^ (((1664525 * ((s & 0xffff0000) >>> 16)) << 16)
                                + 1664525 * (s & 0x0000ffff)))
                        + a[0][j]
                        + j;
                    if (++i >= N) {
                        this.mt[0] = this.mt[N - 1];
                        i = 1;
                    }
                    if (++j >= l) {
                        j = 0;
                    }
                }

                for (var k = N - 1; k != 0; --k) {
                    var s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
                    this.mt[i] = (this.mt[i]
                            ^ (((1566083941 * ((s & 0xffff0000) >>> 16)) << 16)
                                + 1566083941 * (s & 0x0000ffff)))
                        - i;
                    if (++i >= N) {
                        this.mt[0] = this.mt[N - 1];
                        i = 1;
                    }
                }

                this.mt[0] = 0x80000000;
                return;
            default:
                var seeds = new Array();
                for (var i = 0; i < a.length; ++i) {
                    seeds.push(a[i]);
                }
                this.setSeed(seeds);
                return;
        }
    };

    /**
	 * 次の擬似乱数を生成します．
	 * @param bits	出力値の有効ビット数を指定します．
	 * 	0 &lt; bits &lt;= 32で指定します．
	 * @param 次の擬似乱数．
	 */
    FP[$next] = function (bits) {
        if (this.mti >= N) {
            var x = 0;

            for (var k = 0; k < N - M; ++k) {
                x = (this.mt[k] & 0x80000000) | (this.mt[k + 1] & 0x7fffffff);
                this.mt[k] = this.mt[k + M] ^ (x >>> 1) ^ MAG01[x & 0x1];
            }
            for (var k = N - M; k < N - 1; ++k) {
                x = (this.mt[k] & 0x80000000) | (this.mt[k + 1] & 0x7fffffff);
                this.mt[k] = this.mt[k + (M - N)] ^ (x >>> 1) ^ MAG01[x & 0x1];
            }
            x = (this.mt[N - 1] & 0x80000000) | (this.mt[0] & 0x7fffffff);
            this.mt[N - 1] = this.mt[M - 1] ^ (x >>> 1) ^ MAG01[x & 0x1];

            this.mti = 0;
        }

        var y = this.mt[this.mti++];
        y ^= y >>> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >>> 18;
        return y >>> (32 - bits);
    };

    /**
	 * 一様分布のboolean型の擬似乱数を返します．
	 * @return true or false．
	 */
    FP.nextBoolean = function () {
        return this[$next](1) == 1;
    };

    /**
	 * 一様分布の符号無32ビット整数型の擬似乱数を返します．
	 * @return 符号無32ビット整数型の擬似乱数で，0以上4294967295以下です．
	 */
    FP.nextInteger = function () {
        return this[$next](32);
    };

    /**
	 * 一様分布の符号無46ビット整数型の擬似乱数を返します．
	 * @return 符号無46ビット整数型の擬似乱数で，0以上70368744177663以下です．
	 */
    FP.nextLong = function () {
        // NOTE: 48ビット以上で計算結果がくずれる．
        // (46 - 32) = 14 = [7] + [7], 32 - [7] = [25], 32 - [7] = [25]
        // 2^(46 - [25]) = 2^21 = [2097152]
        return this[$next](25) * 2097152 + this[$next](25);
    };

    /**
	 * 0.0～1.0の範囲で一様分布の32ビットベースの
	 * 浮動小数点型の擬似乱数を返します．
	 * @return 半開区間の[0.0 1.0)です．
	 */
    FP.nextFloat = function () {
        return this[$next](32) / 4294967296.0; // 2^32
    };

    /**
	 * 0.0～1.0の範囲で一様分布の46ビットベースの
	 * 浮動小数点型の擬似乱数を返します．
	 * @return 半開区間の[0.0 1.0)です．
	 */
    FP.nextDouble = function () {
        return (this[$next](25) * 2097152 + this[$next](25))
			/ 70368744177664.0; // 2^46
    };

} class__MersenneTwister__(window);



/**
 * 擬似乱数生成にメルセンヌ・ツイスタを使用し，半開区間[0 1.0)の
 * 浮動小数点型の擬似乱数を生成します．
 * Math.randomと同様に使用します．
 * 
 * 使用例)
 * // 0以上1より小さい不動小数点型の値を生成します．
 * var r = Math.mersenneTwisterRandom();
 */
Math.mersenneTwisterRandom = function () {
    Math.__MERSENNE_TWISTER__ = new MersenneTwister();

    return function () {
        return Math.__MERSENNE_TWISTER__.nextFloat();
    }
}();

var theMersenneTwister = new MersenneTwister(0);
var MersenneTwisterIdx = 0;
function Math_random() {
    MersenneTwisterIdx++;
    return theMersenneTwister.nextFloat();
}