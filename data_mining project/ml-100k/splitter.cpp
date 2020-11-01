#include<bits/stdc++.h> 
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp> 
// #pragma GCC optimize("-O3")
// #pragma GCC optimize("Ofast")
// #pragma GCC target("avx,avx2,fma")
using namespace std;
using namespace __gnu_pbds;
#ifdef ON_LINUX  
#include <sys/resource.h>
#define meminc             rlimit rlim;if (getrlimit(RLIMIT_STACK, &rlim)) return 1;rlim.rlim_cur = 26843556;if (setrlimit(RLIMIT_STACK, &rlim)) return 2;
#else
#define meminc      
#endif
#ifdef LOCAL
#define dbg(args...)       { string _s = #args; replace(_s.begin(), _s.end(), ',', ' '); stringstream _ss(_s); istream_iterator<string> _it(_ss); err(_it, args); }
#else
#define dbg(args...)
#endif
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
void err(istream_iterator<string> it) {}
template<typename T, typename... Args>
void err(istream_iterator<string> it, T a, Args... args) {
    cerr << *it << " = " << a << endl;
    err(++it, args...);
}
template<typename T,typename U> std::ostream& operator<<(std::ostream& out, std::pair<T,U> a) {
    out<< a.first << " " << a.second;
    return out;
}
 
template<typename T,typename U> std::istream& operator>>(std::istream& in, std::pair<T,U> &a) {
    in >> a.first >> a.second;
    return in;
}
#define int                long long int
#define ll                 int
#define pb                 push_back
#define pii                pair<int, int>
#define ff                 first
#define se                 second
#define vii                vector<pii>
#define vi                 vector<int >
#define ordered_set        tree<int, null_type,less<int>, rb_tree_tag,tree_order_statistics_node_update>
// -------------------Standard Traversal Moves---------------------
// vi fx = {1 ,-1 ,0, 0}, fy = {0, 0, -1, 1};
// vi fx = {2, -2, 2, -2,  1, -1, 1, -1}, fy = {1, 1, -1, -1, 2, 2, -2, -2};
// vi fx = {1, 1, 1, -1, -1 , -1, 0, 0}, fy = {1, -1, 0, 1, -1, 0, 1, -1};
// ----------------------------------------------------------------

#define rep(i, a, b)       for(int  i=a;i<b;i++)
#define all(a)             (a).begin(),(a).end()
#define sz(x)              (int )x.size()
#define yes                cout << "YES" << endl
#define no                 cout << "NO" << endl
#define endl               '\n' 
// const int hell = (int)998244353;
const int hell = (int)1e9 + 7;
const int inf = 3223372036854775807;
const double PI = 3.14159265;
const int N = (int) 1e6 + 5;
int n, m, k, a[N];
        int counter = 1;
int rand(int a,int b)
{
    return a + (b - a + 1) * rand();
}
int32_t main()
{
  //  meminc;
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cerr.precision(10);
    cout.precision(25);
    cout << fixed;
    #ifdef LOCAL
    // for getting input from input.txt
    freopen("basefile", "r", stdin);
    // for writing output to output.txt
    #endif
    int tests = 1;
    // cin >> tests;
    rep(test, 1, tests+1)
    {
        string filename[2] = {"u90_1.test", "u90_1.base"};
        ofstream outputfile[2];
        rep(i, 0, 2)
        {
            outputfile[i].open(filename[i].c_str());
        }
        n = 100000;
        m = n / 10;
        m *= 9;
        k = n - m;
        srand(20387520947);
        rep(i, 0, n)
        {
            int f1, f2, f3, f4;
            counter++;
            cin >> f1 >> f2 >> f3 >> f4;
            // string g;
            // cin >> g;
            // g += "::";
            // f1 = f2 = f3 = f4 = -1;
            // int le = -1;
            // rep(j, 0, sz(g) - 1)
            // {
            //     if(g[j] == ':' and g[j + 1] == ':')
            //     {
            //         if(f1 == -1)
            //         {
            //             f1 = stoi(g.substr(le + 1, j - le));
            //             continue;
            //         }
            //         else if(f2 == -1)
            //         {
            //             f2 = stoi(g.substr(le + 1, j - le));
            //             continue;
            //         }
            //         else if(f3 == -1)
            //         {
            //             f3 = stoi(g.substr(le + 1, j - le));
            //             continue;
            //         }
            //         else
            //         {
            //             f4 = stoi(g.substr(le + 1, j - le));
            //         }
            //     }
            //     else
            //     {
            //         le = j;
            //     }
            // }
            string to_write;
            to_write += to_string(f1);
            to_write += ',';
            to_write += to_string(f2);
            to_write += ',';
            to_write += to_string(f3);
            to_write += ',';
            to_write += to_string(f4);
            to_write += ',';
            if(!m)
            {
                outputfile[0] << to_write << endl;
                continue;
            }
            if(!k)
            {
                outputfile[1] << to_write << endl;
                continue;
            }
            int key = rand(0, 1e9);
            key &= 1;
            if(!key)
            {
                m--;
                outputfile[1] << to_write << endl;
                continue;
            }
            else{
                k--;
                outputfile[0] << to_write << endl;
            }
        }
    }
    #ifdef LOCAL
        // cerr << "Time elapsed: " << 1.0 * clock() / CLOCKS_PER_SEC << " s.\n";
    #endif
    return 0;
}