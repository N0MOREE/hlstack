// 从 SQD replica-cmds(动作流) + fills(成交) 重放单币种 L4 订单簿。
// 与早期 Python 参考实现(已退役)逐单对齐;快照须与之逐单相等。
//
// 架构:N 个工人线程解压+解析 JSON,当场提炼成紧凑 Op 列表(只留本资产相关的),
//       JSON 树立刻释放;主线程按文件顺序消费 Op,维护簿状态。
//       状态相关的解析(cloid 解析、RO 裁剪)全部留在主线程,保证与 Python 语义一致。
use flate2::read::GzDecoder;
use serde_json::Value;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Write as _};
use std::sync::mpsc;
use std::thread;

// ---------- 紧凑操作(工人 → 主线程) ----------

enum Target { Oid(u64), Cloid(String), None }

struct AddInfo { oid: u64, side: u8, px: f64, sz: f64, cloid: Option<String>, ro: bool,
                 /// modify 换出的新单专用:oid 是块内计数器算出来的而非回执给的。
                 /// true = 前后锚点之间的空位被占号事件正好填满,分配被强制唯一,可证明正确。
                 /// 实测 711 例真值:认证率 92.8%,认证内 0 错。
                 certain: bool }

enum Op {
    Add(AddInfo),
    /// trigger(TP/SL) 单不进可见簿,但它拿到了真 oid 且几乎都是 reduce-only。
    /// 播种快照里会出现这些 oid 而快照不带 ro 字段,所以采收阶段必须收下它们;
    /// 重放阶段忽略。(Python 版的采收循环天然覆盖,这里补齐以保持逐位一致。)
    Meta(AddInfo),
    Bump(&'static str),
    Cancel { oid: u64 },
    CancelCloid { cloid: String },
    /// 撤单被交易所拒了,理由是「Order was never placed, already canceled, or filled」。
    /// 这句话不是"撤单没生效",而是交易所断言这笔单此刻不存在——
    /// 而如果它还在我们簿上,那就是一个确定性矛盾,交易所是权威,我们必须删。
    /// 实测这正是幽灵单的主要来源:交易所自己悄悄删了单,连挂单的地址都不知道,
    /// 它去撤才发现单没了。这个"撤单失败"回执是我们唯一能观测到静默删除的地方。
    /// 与 Cancel 的区别:绝不回退到合成 oid(那是猜),找不到就什么都不做。
    CancelGone { oid: u64 },
    CancelCloidGone { cloid: String },
    Modify { target: Target, add: Option<AddInfo>, trigger: bool },
    BatchItem { target: Target, add: Option<AddInfo>, trigger: bool },
    Sched { time: Option<i64> },
    /// 立即成交的吃单:它不进簿(回执是 filled 而非 resting),所以走不到 add(),
    /// 但它撞上本用户自己的挂单时,交易所同样要做自成交防护。
    /// 只带 side/px,用于跑与 add() 相同的自穿判定。
    Aggr { side: u8, px: f64, sz: f64, oid: u64 },
    /// 下单时立即撮合掉一部分,剩余量仍以同一个 oid 挂在簿上。
    /// 原来的假设「回执是 filled 就说明没进簿」是错的 —— filled.totalSz 只是
    /// 立即成交的那部分。实测 4.85 天有约 550 笔这样的单,而且个头很大
    /// (样本里有下单 200 成交 112、剩 87 挂着的),所以它对金额口径的破坏
    /// 远大于对笔数口径 —— 正好解释了「带内漏金额% 一直远高于漏笔数%」。
    ///
    /// 只认 Gtc:实测 2 万块里 filled 带剩余的 70 笔中,53 笔 Ioc + 15 笔市价,
    /// 这两类的剩余量是取消不是挂住,补挂进去就是凭空造幽灵。真正该补的只有 2 笔 Gtc。
    AddRest(AddInfo),
}

/// user 用 Arc<str>:反序列化时每个 act 都要带上地址,String 克隆是 42 字节堆分配,
/// 实测 6 万块就有 315 万个 act。Arc 克隆只是原子加一。
/// ops 扁平化到块级:原来每个 act 一个 Vec<Op>,又是 315 万次分配。
struct ActC { user: std::sync::Arc<str>, n: u32, ai: u32 }
struct BlockC { n: u64, ts: Option<i64>, acts: Vec<ActC>, ops: Vec<Op>, tags: Vec<u32> }

struct FillC { oid: u64, user: String, sp: Option<f64>, sz: f64, side_b: bool, px: f64, crossed: bool }

// ---------- 簿状态(仅主线程) ----------

/// 块末仲裁的时间基准 —— 一条死亡证明成立于某个时刻,只有比它更早进簿的单才可能被它证死。
///
/// 证据源不同,能给出的时间精度也不同。**精度写在类型里,不藏在 filter 表达式里** ——
/// 这是重构的全部意义:以前三条机制各写一遍 filter,两条用粗尺子一条用细尺子,
/// 而「用的哪把尺子」只有读代码才看得出来。
#[derive(Clone, Copy, Debug)]
enum QBound {
    /// 精确:证据本身是一个**动作**,而动作按 actionIndex 排过序,入簿序号 q 随之严格递增。
    /// 判据 e.q < q_e。resting 回执、未确认单自身都属于这一类。
    Exact(u64),
    /// 粗糙:证据来自**成交流**,而 fills 没有块内序号 —— 只能整块豁免。
    /// 判据退化成「不是本块新增、也不是本块被打中还有余量」。
    /// 一旦拿到吃单方的 actionIndex,这里就能升级成 Exact,判定那一侧一行都不用改。
    BeforeBlock,
}

struct Entry { user: String, side: u8, px: f64, sz: f64, ro: bool,
               /// 入簿序号:同价位内按它排序即得排队顺序(FIFO 假设)
               q: u64 }

/// 落盘流水的一条。ai = 造成这条变更的**动作**在块内的 actionIndex;
/// -1 表示它不由任何单个动作直接造成(成交流扣减、块末仲裁、扫盘清理、
/// scheduleCancel 到期)。有了它,任何一条错误都能反查回原始记录里那一条动作,
/// 而不是只知道"这一块里出的事"。
struct Diff { block: u64, oid: u64, typ: &'static str, why: &'static str, side: u8, px: f64, sz: f64, user: String, ai: i32 }

#[derive(Default)]
struct Book {
    book: HashMap<u64, Entry>,
    cloid: HashMap<(String, String), HashSet<u64>>,
    sched: HashMap<String, i64>,
    pos: HashMap<String, f64>,
    user_oids: HashMap<String, HashSet<u64>>,
    diffs: Vec<Diff>,
    infer_oid: bool,
    rebind_emit: bool,
    ro_proof: bool,
    ro_dedup: bool,
    /// 块内定序:块末仲裁的豁免口径从「同块新增」改为「入簿序号比我晚」。见规则④处的长注释。
    q_order: bool,
    /// 块末自证:簿仍穿价/锁定的块数。不依赖任何外部真值,是引擎的自检指标。
    xblocks: u64,
    /// 前 2000 个穿价现场 (块, 最优买, 最优卖, 买单 oid, 卖单 oid)。
    /// 自证不止要报「还剩几块」,还要报「在哪」—— 否则修不动。微重放台架直接吃这张表。
    xsites: Vec<(u64, f64, f64, u64, u64)>,
    fresh_arbit: bool,
    rule4: bool,
    sweep_clean: bool,
    /// 见 Op::CancelGone 的注释:把「撤单失败=单已不存在」当权威断言执行。
    trust_gone: bool,
    /// 本块内「未经 resting 回执确认」而插入的单(filled-rest 剩余、modify 新单)。
    /// 块末成交处理后逐一检查:限价仍穿过我们对侧最优 → 丢弃。
    /// 依据是撮合引擎的物理定律:一笔单不可能穿着对手盘挂住。只砍未确认入口,
    /// 绝不碰交易所确认过 resting 的单 —— 与已退役的 --uncross 按年龄删单的启发式有本质区别。
    fresh_unconf: Vec<u64>,
    /// 规则④:本块内交易所确认 resting 的 oid。块末逐一检查:若其价格越过
    /// 我们簿上的对侧老单,resting 回执即为老单的死亡证明(撮合引擎不可能让
    /// 两者共存),删老单。豁免口径与 fresh_unconf 仲裁一致(本块新增/本块被打有余量)。
    fresh_conf: Vec<u64>,
    /// 本块内新增的全部 oid。fills 无块内序号,本块后挂的确认单可以合法处于
    /// 扫盘价之内 —— 扫盘清理必须豁免它们(保守方向)。每块末清空。
    fresh_all: HashSet<u64>,
    /// 本块内被成交打中过的 oid。价位在本块有成交且扣完仍有量 = 该价位被证明活着。
    filled_blk: HashSet<u64>,
    /// 立即成交后的 Gtc 剩余量是否补挂进簿。见 Op::AddRest 的注释。
    filled_rest: bool,
    /// 价位 -> 笔数的轻量索引,无条件维护:O(1) 拿到最优买卖价
    /// (fresh_arbit / rule4 / 块末穿价探针在用)。
    pb: BTreeMap<i64, u32>,
    pa: BTreeMap<i64, u32>,
    qn: u64,
    diff_on: bool,   // --no-diffs 时为 false:不再累积 diffs(每条含一次 String 克隆)
    /// 当前正在执行的动作的 actionIndex。主循环在进动作前置位、出动作后复位 -1。
    /// 所有 diff 都从这里取 ai —— 不用把它一路穿过 add/rm 的每个调用点。
    cur_ai: i32,
    st: HashMap<&'static str, u64>,
    // modify 延迟绑定:引擎换 oid 且回执不报告,先用合成 oid 进簿,成交暴露真 oid 后改名
    pending: HashMap<(String, u8, u64), VecDeque<u64>>,   // (user,side,px.to_bits) -> [syn]
    user_syn: HashMap<String, Vec<u64>>,                  // user -> 仍在簿的合成 oid
    oid_ck: HashMap<u64, (String, String)>,               // oid -> cloid 索引键
    syn_next: u64,
}

/// 价格 -> 整数 tick(x100),避免 f64 做 BTreeMap 的键
// 保序精确映射:正 f64 的位模式与数值同序,比较语义 = 真实价格比较。
// 旧版 (px*100).round() 把 <0.005 粒度的价格挤进同一桶:薄币扫盘清理从未生效,
// 规则④更是把同桶当穿价整侧误杀(多币重放时的实测事故)。BTC 整数价不受影响。
fn tick(px: f64) -> i64 { px.to_bits() as i64 }

impl Book {
    /// 块末仲裁的**唯一**判定点。
    ///
    /// 一条死亡证明由三样东西定义:
    ///   `kill_side`  要清理哪一侧
    ///   `bound`      价格边界(该侧越过它的单受影响);`strict` 决定含不含等于
    ///   `qb`         时间基准 —— 证据成立的那一刻
    /// 判定:该侧 ∧ 越界 ∧ 比证据更早进簿 → 死。
    ///
    /// 三个证据源只是三个适配器:
    ///   ① resting 回执   → bound = 该单价格, qb = Exact(该单的 q)
    ///   ② 成交流扫价     → bound = 成交价,   qb = BeforeBlock(fills 无块内序号)
    ///   ③ 簿自身穿价     → bound = 穿价档,   qb = Exact(未确认单的 q)
    ///
    /// `why_base` / `why_margin` 分开记账:前者是粗口径本来也会杀掉的,
    /// 后者是**只有靠块内定序才杀得到**的(粗口径会豁免它)。
    /// 这样 q-order 的边际杀伤在落盘流水里可以直接数出来。
    fn arbitrate(&mut self, blk: u64, kill_side: u8, bound: i64, strict: bool,
                 qb: QBound, skip: Option<u64>,
                 why_base: &'static str, why_margin: &'static str,
                 counter: Option<&'static str>) {
        let dead: Vec<(u64, bool)> = self.book.iter()
            .filter(|(o, e)| {
                if Some(**o) == skip || e.side != kill_side { return false }
                let t = tick(e.px);
                let in_range = if kill_side == b'B' {
                    if strict { t > bound } else { t >= bound }
                } else if strict { t < bound } else { t <= bound };
                if !in_range { return false }
                // 粗口径:不是本块新增、也不是本块被打中还有余量
                let coarse = !self.fresh_all.contains(o)
                             && !(self.filled_blk.contains(o) && e.sz > 1e-9);
                match qb {
                    QBound::Exact(q_e) => e.q < q_e,
                    QBound::BeforeBlock => coarse,
                }
            })
            .map(|(o, e)| {
                let coarse = !self.fresh_all.contains(o)
                             && !(self.filled_blk.contains(o) && e.sz > 1e-9);
                (*o, !coarse)          // true = 粗口径会豁免它,是块内定序的边际杀伤
            })
            .collect();
        for (o, only_q) in dead {
            if let Some(c) = counter { self.bump(c); }
            self.rm(blk, o, if only_q { why_margin } else { why_base });
        }
    }

    fn bump(&mut self, k: &'static str) { *self.st.entry(k).or_insert(0) += 1; }

    /// rm 的计数器名。原实现是 format!("remove_{}", why),每次增删都堆分配一次;
    /// why 只有 10 个编译期已知的取值,查表换成 &'static str。
    /// 若将来新增 why 而忘了加进这张表,difftest 的计数器差分会立刻抓到。
    /// 新增类机制的计数器名(与 diff 的 why 同名,便于统计对齐)
    fn add_key(src: &str) -> &'static str {
        match src {
            "add" => "add_plain",
            "add_modify" => "add_modify",
            "add_modsyn" => "add_modsyn",
            "add_batch" => "add_batch",
            "add_filledrest" => "add_filledrest",
            "rebind" => "add_rebind",
            _ => "add_UNKNOWN",
        }
    }

    fn rm_key(why: &str, hit: bool) -> &'static str {
        match (why, hit) {
            ("fill", true) => "remove_fill",              ("fill", false) => "rm_miss_fill",
            ("cancel", true) => "remove_cancel",          ("cancel", false) => "rm_miss_cancel",
            ("cancel_syn", true) => "remove_cancel_syn",  ("cancel_syn", false) => "rm_miss_cancel_syn",
            ("cancelByCloid", true) => "remove_cancelByCloid", ("cancelByCloid", false) => "rm_miss_cancelByCloid",
            ("batchModify", true) => "remove_batchModify", ("batchModify", false) => "rm_miss_batchModify",
            ("modify", true) => "remove_modify",          ("modify", false) => "rm_miss_modify",
            ("modify_syn", true) => "remove_modify_syn",  ("modify_syn", false) => "rm_miss_modify_syn",
            ("sched", true) => "remove_sched",            ("sched", false) => "rm_miss_sched",
            ("ro_flat", true) => "remove_ro_flat",        ("ro_flat", false) => "rm_miss_ro_flat",
            ("ro_trim", true) => "remove_ro_trim",        ("ro_trim", false) => "rm_miss_ro_trim",
            ("ro_dup", true) => "remove_ro_dup",      ("ro_dup", false) => "rm_miss_ro_dup",
            ("aggr_drop", true) => "remove_aggr_drop",    ("aggr_drop", false) => "rm_miss_aggr_drop",
            ("swept", true) => "remove_swept",            ("swept", false) => "rm_miss_swept",
            ("stale_cross", true) => "remove_stale_cross", ("stale_cross", false) => "rm_miss_stale_cross",
            // 拆开:rule4 的死亡证明 / 块末穿价仲裁 / q-order 带来的边际杀伤
            ("stale_cross_r4", true) => "remove_stale_cross_r4", ("stale_cross_r4", false) => "rm_miss_stale_cross_r4",
            ("stale_cross_q", true) => "remove_stale_cross_q",   ("stale_cross_q", false) => "rm_miss_stale_cross_q",
            ("stale_cross_fa", true) => "remove_stale_cross_fa", ("stale_cross_fa", false) => "rm_miss_stale_cross_fa",
            ("gone", true) => "remove_gone",              ("gone", false) => "rm_miss_gone",
            (_, true) => "remove_UNKNOWN",                (_, false) => "rm_miss_UNKNOWN",
        }
    }

    /// RO 重挂判定:同一用户在同一 (方向, 价, 量) 上已有 resting 的 reduce-only 单,
    /// 又来一张一模一样的 —— 理性交易者不会同时持有两张完全相同的减仓单,
    /// 且交易所对 RO 单按仓位做总量钳制,两张同样的单只有在仓位 ≥ 2 倍时才可能共存。
    /// 判为「重挂替换」:删掉旧的那张(们)。证据强度弱于回执类规则,故单独开关并对真值验证。
    fn ro_dedup_scan(&mut self, blk: u64, user: &str, side: u8, px: f64, sz: f64) {
        let t = tick(px);
        let mut dup: Vec<u64> = match self.user_oids.get(user) {
            None => Vec::new(),
            Some(set) => set.iter().filter(|o| match self.book.get(o) {
                None => false,
                Some(e) => e.ro && e.side == side && tick(e.px) == t
                           && (e.sz - sz).abs() < 1e-12,
            }).cloned().collect(),
        };
        dup.sort_unstable();
        for o in dup { self.rm(blk, o, "ro_dup"); }
    }

    /// src = 这张单是**哪条机制**建的。写进 diff 的 why,让落盘流水自带来源标签 ——
    /// 「受影响的 oid 存档」靠它,不用另建索引。
    fn add(&mut self, blk: u64, user: &str, a: AddInfo, src: &'static str) {
        // 只在仓位未知时启用:仓位已知时 ro_trim 用真实上限精确裁剪,轮不到启发式
        if self.ro_dedup && a.ro && !self.pos.contains_key(user) {
            self.ro_dedup_scan(blk, user, a.side, a.px, a.sz);
        }
        self.qn += 1;
        let qn = self.qn;
        self.book.insert(a.oid, Entry { user: user.to_string(), side: a.side,
                                        px: a.px, sz: a.sz, ro: a.ro, q: qn });
        self.fresh_all.insert(a.oid);
        self.user_oids.entry(user.to_string()).or_default().insert(a.oid);
        if let Some(c) = &a.cloid {
            self.cloid.entry((user.to_string(), c.clone())).or_default().insert(a.oid);
            self.oid_ck.insert(a.oid, (user.to_string(), c.clone()));
        }
        {
            // pb/pa 无条件维护:穿价检查需要随时知道对侧最优(开销为 BTreeMap 计数,可忽略)
            let m = if a.side == b'B' { &mut self.pb } else { &mut self.pa };
            *m.entry(tick(a.px)).or_insert(0) += 1;
        }
        if self.diff_on {
            self.diffs.push(Diff { block: blk, oid: a.oid, typ: "new", why: src, side: a.side,
                                   px: a.px, sz: a.sz, user: user.to_string(), ai: self.cur_ai });
        }
        self.bump("new"); self.bump(Book::add_key(src));
    }

    fn rm(&mut self, blk: u64, oid: u64, why: &str) {
        match self.book.remove(&oid) {
            None => { let k = Book::rm_key(why, false); self.bump(k) }
            Some(e) => {
                if let Some(s) = self.user_oids.get_mut(&e.user) { s.remove(&oid); }
                {
                    let t = tick(e.px);
                    let m = if e.side == b'B' { &mut self.pb } else { &mut self.pa };
                    if let Some(c) = m.get_mut(&t) {
                        *c -= 1;
                        if *c == 0 { m.remove(&t); }
                    }
                }
                if self.diff_on {
                    self.diffs.push(Diff { block: blk, oid, typ: "remove", why: Book::rm_key(why, true), side: e.side,
                                           px: e.px, sz: e.sz, user: e.user, ai: self.cur_ai });
                }
                let k = Book::rm_key(why, true); self.bump(k);
            }
        }
    }

    fn new_syn(&mut self) -> u64 {
        if self.syn_next == 0 { self.syn_next = u64::MAX; } else { self.syn_next -= 1; }
        self.syn_next
    }

    /// modify/cancel 指向我们不认识的 oid 时,它多半就是本用户上一次 modify 造出的那笔
    fn take_syn(&mut self, user: &str, side: Option<u8>) -> Option<u64> {
        let lst = match self.user_syn.get_mut(user) { Some(l) => l, None => return None };
        let mut i = lst.len();
        while i > 0 {
            i -= 1;
            let o = lst[i];
            match self.book.get(&o) {
                None => { lst.remove(i); }
                Some(e) => {
                    if side.is_some() && Some(e.side) != side { continue; }
                    lst.remove(i); return Some(o);
                }
            }
        }
        None
    }

    /// 把合成 oid 改成引擎的真 oid
    fn rebind(&mut self, blk: u64, syn: u64, real: u64) -> bool {
        let e = match self.book.remove(&syn) { Some(e) => e, None => return false };
        let user = e.user.clone();
        if self.rebind_emit && self.diff_on {
            // 认亲公示:合成号退场 + 真号入场,流从此自洽
            self.diffs.push(Diff { block: blk, oid: syn, typ: "remove", why: "remove_rebind",
                                   side: e.side, px: e.px, sz: e.sz, user: user.clone(), ai: self.cur_ai });
            self.diffs.push(Diff { block: blk, oid: real, typ: "new", why: "rebind",
                                   side: e.side, px: e.px, sz: e.sz, user: user.clone(), ai: self.cur_ai });
            self.bump("rebind_emitted");
        }
        self.book.insert(real, e);
        if let Some(s) = self.user_oids.get_mut(&user) { s.remove(&syn); s.insert(real); }
        if let Some(ck) = self.oid_ck.remove(&syn) {
            if let Some(s) = self.cloid.get_mut(&ck) { s.remove(&syn); s.insert(real); }
            self.oid_ck.insert(real, ck);
        }
        if let Some(l) = self.user_syn.get_mut(&user) { l.retain(|x| *x != syn); }
        if let Some(x) = self.fresh_unconf.iter_mut().find(|x| **x == syn) { *x = real; }
        self.bump("rebind_ok");
        true
    }

    fn resolve_target(&self, user: &str, t: &Target) -> Vec<u64> {
        match t {
            Target::Cloid(s) => match self.cloid.get(&(user.to_string(), s.clone())) {
                Some(oids) => {
                    let mut v: Vec<u64> =
                        oids.iter().filter(|o| self.book.contains_key(o)).cloned().collect();
                    v.sort_unstable();
                    v
                }
                None => vec![],
            },
            Target::Oid(o) => vec![*o], // 与 py 一致:数字目标无论在不在簿都返回
            Target::None => vec![],
        }
    }

    fn ro_trim(&mut self, blk: u64, user: &str) {
        let pos = match self.pos.get(user) {
            Some(p) => *p,
            None => {
                // RO 单能 resting 本身就是交易所认证「此刻有仓」;仓位未知一律不删
                if self.ro_proof { self.bump("ro_unknown_skip"); }
                return;
            }
        };
        let side: Option<u8> =
            if pos > 0.0 { Some(b'A') } else if pos < 0.0 { Some(b'B') } else { None };
        let mut ros: Vec<(u64, u8, f64)> = self.user_oids.get(user).map(|s| {
            s.iter().filter_map(|o| self.book.get(o)
                .filter(|e| e.ro).map(|e| (*o, e.side, e.sz))).collect()
        }).unwrap_or_default();
        ros.sort_unstable_by_key(|x| x.0); // oid 升序 = 最老在前
        match side {
            None => { for (o, _, _) in ros { self.rm(blk, o, "ro_flat"); } }
            Some(sd) => {
                let (same, other): (Vec<_>, Vec<_>) = ros.into_iter().partition(|x| x.1 == sd);
                for (o, _, _) in other { self.rm(blk, o, "ro_flat"); }
                let mut total: f64 = same.iter().map(|x| x.2).sum();
                let cap = pos.abs() + 1e-9;
                let mut i = 0usize;
                while total > cap && i < same.len() {
                    let (o, _, sz) = same[i]; total -= sz; i += 1;
                    self.rm(blk, o, "ro_trim");
                }
            }
        }
    }
}

// ---------- 提炼:JSON → Op(工人线程) ----------

fn fval(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn extract_add(od: &Value, st_obj: &serde_json::Map<String, Value>) -> AddInfo {
    AddInfo {
        oid: st_obj["resting"]["oid"].as_u64().unwrap(),
        side: if od["b"].as_bool() == Some(true) { b'B' } else { b'A' },
        px: fval(&od["p"]),
        sz: fval(&od["s"]),
        cloid: od["c"].as_str().map(|s| s.to_string()),
        ro: od["r"].as_bool() == Some(true),
        certain: true,
    }
}

/// 回执状态里的 oid（resting 或 filled 都算),没有就是 None。
fn st_oid(s: &Value) -> Option<u64> {
    for k in ["resting", "filled"] {
        if let Some(o) = s.get(k).and_then(|x| x.get("oid")).and_then(|x| x.as_u64()) {
            return Some(o);
        }
    }
    None
}

/// 取限价单的 tif（Gtc / Ioc / Alo / FrontendMarket）。触发单没有这一层。
fn tif_of(od: &Value) -> Option<&str> {
    od.get("t")?.get("limit")?.get("tif")?.as_str()
}

/// 撤单回执是否等于「交易所断言这笔单已不存在」。
/// 只认这一句:另外的失败理由(参数错、权限、限频)都不构成对簿状态的断言。
/// 尾部还带 " asset=N",所以用前缀匹配。
fn gone(s: Option<&Value>) -> bool {
    match s {
        Some(Value::String(t)) =>
            t.starts_with("Order was never placed, already canceled, or filled"),
        Some(Value::Object(m)) => m.get("error").and_then(|e| e.as_str())
            .map_or(false, |t| t.starts_with("Order was never placed, already canceled, or filled")),
        _ => false,
    }
}

fn is_trigger(od: &Value) -> bool {
    od["t"].as_object().map_or(false, |t| t.contains_key("trigger"))
}

fn extract_target(v: Option<&Value>) -> Target {
    match v {
        Some(Value::String(s)) => Target::Cloid(s.clone()),
        Some(x) if x.is_u64() => Target::Oid(x.as_u64().unwrap()),
        _ => Target::None,
    }
}

const TAG_ALL: u32 = u32::MAX;    // 账户级动作(scheduleCancel):广播到所有簿

fn extract_block(rec: &Value, assets: &std::collections::HashMap<i64, u32>, stp_modify_err: bool) -> Option<BlockC> {
    let n = rec["header"]["number"].as_u64()?;
    let ts = rec["header"]["timestamp"].as_i64();
    let empty = vec![];
    let mut acts: Vec<&Value> = rec["actions"].as_array().unwrap_or(&empty).iter().collect();
    acts.sort_by_key(|a| a["actionIndex"].as_u64().unwrap_or(0));

    // ---------- 预扫:重建块内的 oid 计数器 ----------
    // oid 是全交易所共用的计数器,严格按 actionIndex 顺序发放。modify 的回执是
    // {"type":"default"},不含新单 oid —— 但它一定占用了它那个位置上的号,所以能算出来。
    //
    // 两个反直觉的前提(都是实测出来的,少任何一个都对不齐):
    //   ① 必须数全部币种。250 多个币共用一个计数器,只看 BTC 的话每块
    //      中位有 173 个"缺口"永远解释不了。
    //   ② 被拒的下单腿也占号。BTC 订单腿 93% 被拒,每块约 230 笔 post-only
    //      越价被拒 —— 正好对上每块 222 个缺口。它们拿了号才被撮合弹回来。
    //      唯一的例外是「Order has invalid price」:那是参数校验阶段就打回,
    //      还没轮到分配编号。(4 个反例全由它造成,排除后归零。)
    //
    // 认证:本条 modify 前后两个锚点之间有几个空位,中间就有几个占号事件 ——
    // 正好填满,分配就是被强制的,不是猜的。实测 711 例:认证 92.8%,认证内 0 错。
    let mut pred: std::collections::HashMap<(usize, usize), (u64, bool)> =
        std::collections::HashMap::new();
    {
        let mut seq: Vec<(bool, u64)> = Vec::new();             // (是锚点, 锚点值)
        let mut slots: Vec<(usize, usize, usize)> = Vec::new(); // (seq下标, act下标, leg下标)
        let empty2: Vec<Value> = vec![];
        for (ai, act) in acts.iter().enumerate() {
            let ac = &act["action"];
            let typ = ac["type"].as_str().unwrap_or("");
            let sts = act["response"].get("data").and_then(|d| d.get("statuses"))
                .and_then(|s| s.as_array());
            match typ {
                "order" => {
                    if let Some(ss) = sts {
                        for st in ss {
                            if let Some(o) = st_oid(st) { seq.push((true, o)); }
                            else if let Some(e) = st.get("error").and_then(|x| x.as_str()) {
                                if !e.starts_with("Order has invalid price") {
                                    seq.push((false, 0));
                                }
                            }
                        }
                    }
                }
                "modify" | "batchModify" => {
                    let legs: Vec<&Value> = if typ == "modify" { vec![ac] }
                        else { ac["modifies"].as_array().unwrap_or(&empty2).iter().collect() };
                    for (i, m) in legs.iter().enumerate() {
                        let od = &m["order"];
                        if od.is_null() { continue; }
                        let st = sts.and_then(|x| x.get(i));
                        if let Some(sv) = st {
                            if sv.get("error").is_some() { continue; }   // 被拒的 modify 不占号
                            if let Some(o) = st_oid(sv) { seq.push((true, o)); continue; }
                        }
                        if od["a"].as_i64().map_or(false, |a| assets.contains_key(&a)) { slots.push((seq.len(), ai, i)); }
                        seq.push((false, 0));
                    }
                }
                _ => {}
            }
        }
        for (p, ai, li_) in slots {
            let (mut lo, mut li) = (None, 0usize);
            for j in (0..p).rev() { if seq[j].0 { lo = Some(seq[j].1); li = j; break; } }
            let (mut hi, mut hj) = (None, 0usize);
            for j in (p + 1)..seq.len() { if seq[j].0 { hi = Some(seq[j].1); hj = j; break; } }
            if let (Some(lo), Some(hi)) = (lo, hi) {
                if hi <= lo { continue; }
                let between = hj - li - 1;
                let room = (hi - lo - 1) as usize;
                let mypos = p - li - 1;
                let certain = between == room && mypos < room;
                pred.insert((ai, li_), (lo + 1 + mypos as u64, certain));
            }
        }
    }

    let mut out: Vec<ActC> = Vec::new();
    let mut flat: Vec<Op> = Vec::new();
    let mut tags: Vec<u32> = Vec::new();
    for (act_i, act) in acts.into_iter().enumerate() {
        if act["status"].as_str() != Some("ok") {
            // 第九条语义:modify = 先撤后挂。回执 "Error placing new order during modify: ..."
            // 说明撤单腿已经执行成功、只有挂新单那一步失败 —— 原单在交易所侧已经没了。
            // 原实现对 status != ok 一律整条跳过,于是原单永远留在我们簿上,成为幽灵。
            // 只匹配这一句前缀:另外两类("Cannot modify canceled or filled order" /
            // "Order price cannot be more than 95% away") 是整条被拒,跳过才是对的。
            if stp_modify_err && act["action"]["type"].as_str() == Some("modify") {
                if act["response"].as_str()
                    .map_or(false, |r| r.starts_with("Error placing new order during modify")) {
                    let ac = &act["action"];
                    if let Some(&bi) = ac["order"]["a"].as_i64()
                            .and_then(|a| assets.get(&a)) {
                        let mut ops: Vec<(u32, Op)> = Vec::new();
                        ops.push((bi, Op::Bump("modify_err_cancel")));
                        match extract_target(ac.get("oid")) {
                            Target::Oid(o) => ops.push((bi, Op::Cancel { oid: o })),
                            Target::Cloid(c) => ops.push((bi, Op::CancelCloid { cloid: c })),
                            Target::None => {}
                        }
                        let user = act["user"].as_str().unwrap_or("");
                        out.push(ActC { user: std::sync::Arc::from(user), n: ops.len() as u32,
                                        ai: act["actionIndex"].as_u64().unwrap_or(0) as u32 });
                        for (t, o) in ops { tags.push(t); flat.push(o); }
                    }
                }
            }
            continue;
        }
        let ac = &act["action"];
        let typ = ac["type"].as_str().unwrap_or("");
        let user = act["user"].as_str().unwrap_or("");
        let sts = act["response"].get("data").and_then(|d| d.get("statuses"))
            .and_then(|s| s.as_array());
        let mut ops: Vec<(u32, Op)> = Vec::new();

        match typ {
            "order" => {
                if let Some(ods) = ac["orders"].as_array() {
                    for (i, od) in ods.iter().enumerate() {
                        let bi = match od["a"].as_i64().and_then(|a| assets.get(&a)) {
                            Some(&x) => x, None => continue };
                        match sts.and_then(|s| s.get(i)) {
                            Some(Value::Object(m)) if m.contains_key("resting") => {
                                if is_trigger(od) {
                                    ops.push((bi, Op::Bump("trigger_skipped")));
                                    ops.push((bi, Op::Meta(extract_add(od, m))));
                                } else { ops.push((bi, Op::Add(extract_add(od, m)))); }
                            }
                            Some(Value::Object(m)) if m.contains_key("error") =>
                                ops.push((bi, Op::Bump("order_rejected"))),
                            Some(Value::Object(m)) if m.contains_key("filled") => {
                                ops.push((bi, Op::Bump("order_filled_immediately")));
                                let side = if od["b"].as_bool() == Some(true) { b'B' } else { b'A' };
                                let toid = m["filled"].get("oid").and_then(|v| v.as_u64()).unwrap_or(0);
                                ops.push((bi, Op::Aggr { side, px: fval(&od["p"]), sz: fval(&od["s"]), oid: toid }));
                                // 剩余量仍挂在簿上 —— 但只有 Gtc 是这样,
                                // Ioc / 市价单的剩余是当场取消,补挂就是造幽灵。
                                if let Some(fm) = m["filled"].as_object() {
                                    let rem = fval(&od["s"]) - fval(&fm["totalSz"]);
                                    if rem > 1e-9 && tif_of(od) == Some("Gtc") && !is_trigger(od) {
                                        if let Some(oid) = fm.get("oid").and_then(|v| v.as_u64()) {
                                            // 插全量,同块成交会按 oid 扣掉 totalSz,
                                            // 剩余自动正确 —— 单一事实来源,不再手算差额
                                            // (手算差额 + 成交再扣 = 双重扣减,实测抓到)。
                                            ops.push((bi, Op::AddRest(AddInfo {
                                                oid, side, px: fval(&od["p"]), sz: fval(&od["s"]),
                                                cloid: od["c"].as_str().map(|s| s.to_string()),
                                                ro: od["r"].as_bool() == Some(true),
                                                certain: true })));
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "cancel" => {
                if let Some(cs) = ac["cancels"].as_array() {
                    for (i, c) in cs.iter().enumerate() {
                        let bi = match c["a"].as_i64().and_then(|a| assets.get(&a)) {
                            Some(&x) => x, None => continue };
                        let ok = match sts.and_then(|s| s.get(i)) {
                            None => true,
                            Some(Value::String(s)) => s == "success",
                            _ => false,
                        };
                        if ok { ops.push((bi, Op::Cancel { oid: c["o"].as_u64().unwrap_or(0) })); }
                        else if gone(sts.and_then(|s| s.get(i))) {
                            ops.push((bi, Op::CancelGone { oid: c["o"].as_u64().unwrap_or(0) }));
                        }
                        else { ops.push((bi, Op::Bump("cancel_failed"))); }
                    }
                }
            }
            "cancelByCloid" => {
                if let Some(cs) = ac["cancels"].as_array() {
                    for (i, c) in cs.iter().enumerate() {
                        let bi = match c["asset"].as_i64().and_then(|a| assets.get(&a)) {
                            Some(&x) => x, None => continue };
                        let ok = match sts.and_then(|s| s.get(i)) {
                            None => true,
                            Some(Value::String(s)) => s == "success",
                            _ => false,
                        };
                        if ok { ops.push((bi, Op::CancelCloid {
                            cloid: c["cloid"].as_str().unwrap_or("").to_string() })); }
                        else if gone(sts.and_then(|s| s.get(i))) {
                            ops.push((bi, Op::CancelCloidGone {
                                cloid: c["cloid"].as_str().unwrap_or("").to_string() }));
                        }
                        else { ops.push((bi, Op::Bump("cancelcloid_failed"))); }
                    }
                }
            }
            "modify" => {
                let od = &ac["order"];
                if let Some(&bi) = od["a"].as_i64().and_then(|a| assets.get(&a)) {
                    let trig = is_trigger(od);
                    let (po, pc) = pred.get(&(act_i, 0)).copied().unwrap_or((0, false));
                    let add = if trig { None } else { Some(AddInfo {
                        oid: po,                    // 0 = 没算出来,主线程退合成 oid
                        side: if od["b"].as_bool() == Some(true) { b'B' } else { b'A' },
                        px: fval(&od["p"]), sz: fval(&od["s"]),
                        cloid: od["c"].as_str().map(|s| s.to_string()),
                        ro: od["r"].as_bool() == Some(true),
                        certain: pc,
                    }) };
                    ops.push((bi, Op::Modify { target: extract_target(ac.get("oid")), add, trigger: trig }));
                }
            }
            "batchModify" => {
                if let Some(mods) = ac["modifies"].as_array() {
                    for (i, m) in mods.iter().enumerate() {
                        let od = &m["order"];
                        let bi = match od["a"].as_i64().and_then(|a| assets.get(&a)) {
                            Some(&x) => x, None => continue };
                        let target = extract_target(m.get("oid"));
                        let mut meta = None;
                        let (add, trigger) = match sts.and_then(|s| s.get(i)) {
                            Some(Value::Object(mm)) if mm.contains_key("resting") => {
                                if is_trigger(od) { meta = Some(extract_add(od, mm)); (None, true) }
                                else { (Some(extract_add(od, mm)), false) }
                            }
                            _ => (None, false),
                        };
                        ops.push((bi, Op::BatchItem { target, add, trigger }));
                        if let Some(a) = meta { ops.push((bi, Op::Meta(a))); }
                    }
                }
            }
            "scheduleCancel" => ops.push((TAG_ALL, Op::Sched { time: ac["time"].as_i64() })),
            _ => {}
        }
        if !ops.is_empty() {
            out.push(ActC { user: std::sync::Arc::from(user), n: ops.len() as u32,
                            ai: act["actionIndex"].as_u64().unwrap_or(0) as u32 });
            for (t, o) in ops { tags.push(t); flat.push(o); }
        }
    }
    Some(BlockC { n, ts, acts: out, ops: flat, tags })
}

// ---------- 文件与流水线 ----------

fn sorted_gz_files(dir: &str) -> Vec<std::path::PathBuf> {
    let mut fs: Vec<_> = std::fs::read_dir(dir).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".ndjson.gz"))
        .collect();
    fs.sort_by_key(|p| {
        p.file_name().unwrap().to_string_lossy()
            .split('-').next().unwrap().parse::<u64>().unwrap_or(0)
    });
    fs
}

#[allow(clippy::type_complexity)]   // 返回 (rx, 工人句柄) 元组,只此一处调用,不值得起类型别名
fn parse_pipeline(files: Vec<std::path::PathBuf>, workers: usize,
    assets: std::sync::Arc<std::collections::HashMap<i64, u32>>, mod_err: bool)
    -> (mpsc::Receiver<(usize, Vec<BlockC>)>, Vec<thread::JoinHandle<()>>) {
    let (tx, rx) = mpsc::sync_channel::<(usize, Vec<BlockC>)>(workers * 2);
    let jobs: Vec<(usize, std::path::PathBuf)> = files.into_iter().enumerate().collect();
    let jobs = std::sync::Arc::new(std::sync::Mutex::new(jobs.into_iter()));
    // 把 JoinHandle 交给主线程:工人 panic(坏 gz/坏 JSON 的 unwrap)只会关掉它那份 tx,
    // 主循环的 recv Err 分支会把这当成正常读完 —— 必须靠 join 取回工人结果才能发现截断。
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let tx = tx.clone();
        let jobs = jobs.clone();
        let assets = assets.clone();
        let mod_err = mod_err;
        handles.push(thread::spawn(move || loop {
            let job = { jobs.lock().unwrap().next() };
            let (idx, path) = match job { Some(j) => j, None => break };
            let rd = BufReader::with_capacity(1 << 20, GzDecoder::new(File::open(&path).unwrap()));
            let mut out = Vec::new();
            for line in rd.lines() {
                let line = line.unwrap();
                if line.trim().is_empty() { continue; }
                let v: Value = serde_json::from_str(&line).unwrap();
                if let Some(bc) = extract_block(&v, &assets, mod_err) { out.push(bc); }
                // v 在此处释放:JSON 树不出工人线程
            }
            if tx.send((idx, out)).is_err() { break; }
        }));
    }
    (rx, handles)
}

// ---------- 主流程 ----------

fn main() {
    let mut args: HashMap<String, String> = HashMap::default();
    let argv: Vec<String> = std::env::args().collect();
    // 合法 flag 全集(与下文所有 args.get 一一对应)。新增 flag 时必须同步这张表,
    // 否则会被当成未知参数拒收 —— 这正是要的效果:拼错/已删除的开关不再被静默吞掉。
    const KNOWN_FLAGS: &[&str] = &[
        "--data", "--asset", "--coin", "--warmup-blocks", "--modify-err-cancel",
        "--emit-from", "--to-block", "--out", "--multi", "--seed-dir", "--seed-block",
        "--snap-at", "--synthetic-modify", "--workers", "--no-diffs", "--trust-gone",
        "--filled-rest", "--infer-oid", "--fresh-arbit", "--rule4", "--sweep-clean",
        "--rebind-emit", "--ro-proof", "--ro-dedup", "--q-order", "--taker-tape",
    ];
    let mut i = 1;
    while i + 1 < argv.len() {
        if argv[i].starts_with("--") && !KNOWN_FLAGS.contains(&argv[i].as_str()) {
            eprintln!("未知参数 {}", argv[i]);
            std::process::exit(2);
        }
        args.insert(argv[i].clone(), argv[i + 1].clone());
        i += 2;
    }
    // 成对步进剩下的孤参数(flag 缺值)同样拒收,不再静默丢弃
    if i < argv.len() && argv[i].starts_with("--") {
        eprintln!("未知参数或缺少取值: {}", argv[i]);
        std::process::exit(2);
    }
    let data = args.get("--data").cloned().unwrap_or("data".into());
    let asset: i64 = args.get("--asset").map(|s| s.parse().unwrap()).unwrap_or(0);
    let coin = args.get("--coin").cloned().unwrap_or("BTC".into());
    let warmup: u64 = args.get("--warmup-blocks").map(|s| s.parse().unwrap()).unwrap_or(42840);
    // --emit-from 之前是热身区:簿子残缺(缺窗口之前挂的单),不出快照不对账。--to-block 终点。
    let mod_err: bool = args.get("--modify-err-cancel").map(|s| s == "1").unwrap_or(false);
    let emit_from: u64 = args.get("--emit-from").map(|s| s.parse().unwrap()).unwrap_or(0);
    let to_block: u64 = args.get("--to-block").map(|s| s.parse().unwrap()).unwrap_or(u64::MAX);
    let out = args.get("--out").cloned().unwrap_or("replay_out_rs".into());
    // --multi <文件>:每行 "<asset> <coin>",一次重放多个币。缺省=单币(旧行为逐位不变)
    let multi: Vec<(i64, String)> = match args.get("--multi") {
        Some(p) => std::fs::read_to_string(p).expect("读不到 --multi 文件").lines()
            .filter_map(|l| { let mut it = l.split_whitespace();
                Some((it.next()?.parse().ok()?, it.next()?.to_string())) }).collect(),
        None => vec![(asset, coin.clone())],
    };
    let nb: usize = multi.len();
    let assets_map: std::collections::HashMap<i64, u32> =
        multi.iter().enumerate().map(|(i, (a, _))| (*a, i as u32)).collect();
    let coins: Vec<String> = multi.iter().map(|(_, c)| c.clone()).collect();
    let coin2bi: std::collections::HashMap<String, u32> =
        multi.iter().enumerate().map(|(i, (_, c))| (c.clone(), i as u32)).collect();
    let seed_dir = args.get("--seed-dir").cloned();
    if nb > 1 { eprintln!("多币模式: {} 个簿 {:?}", nb, coins); }
    let seed_block: u64 = args.get("--seed-block").map(|s| s.parse().unwrap()).unwrap_or(0);
    // --snap-at <文件>:每行一个毫秒时间戳,引擎走到该时刻就落盘。
    // 用固定网格(绝对整点)而不是"从播种块起每 N 毫秒",换播种块/换数据都落在同一批瞬间,
    // 于是商业真值买一次就能被所有版本复用,不同版本的曲线也能逐点相减。
    let snap_at: Vec<i64> = args.get("--snap-at").map(|p| {
        let mut v: Vec<i64> = std::fs::read_to_string(p).unwrap().lines()
            .filter_map(|l| l.trim().parse().ok()).collect();
        v.sort_unstable(); v.dedup(); v
    }).unwrap_or_default();
    let mut snap_i: usize = 0;
    if !snap_at.is_empty() {
        eprintln!("固定快照网格: {} 个时刻, {} → {}", snap_at.len(), snap_at[0], snap_at[snap_at.len()-1]);
    }
    // 合成 oid(modify 换号时按块内计数器推算新 oid)实测净负,默认关。
    // 取值式而非裸开关:上面的 argv 解析是严格成对步进的,裸 flag 会错位后续参数。
    let syn_mod: bool = args.get("--synthetic-modify")
        .map(|s| s == "1" || s == "true").unwrap_or(false);
    let workers: usize = args.get("--workers").map(|s| s.parse().unwrap())
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    std::fs::create_dir_all(&out).unwrap();
    eprintln!("workers = {}", workers);

    // 预读 fills(体积小),按块索引成紧凑结构
    let mut fills_by_block: HashMap<u64, Vec<(u32, FillC)>> = HashMap::default();
    let mut nfills = 0u64;
    for f in sorted_gz_files(&format!("{}/fills", data)) {
        let rd = BufReader::with_capacity(1 << 20, GzDecoder::new(File::open(&f).unwrap()));
        for line in rd.lines() {
            let line = line.unwrap();
            if line.trim().is_empty() { continue; }
            let v: Value = serde_json::from_str(&line).unwrap();
            let n = match v["header"]["number"].as_u64() { Some(n) => n, None => continue };
            if let Some(recs) = v["fills"].as_array() {
                for r in recs {
                    let bi = match r["coin"].as_str().and_then(|c| coin2bi.get(c)) {
                        Some(&x) => x, None => continue };
                    fills_by_block.entry(n).or_default().push((bi, FillC {
                        oid: r["oid"].as_u64().unwrap_or(0),
                        user: r["user"].as_str().unwrap_or("").to_string(),
                        sp: if r["startPosition"].is_null() { None }
                            else { Some(fval(&r["startPosition"])) },
                        sz: fval(&r["sz"]).abs(),
                        side_b: r["side"].as_str() == Some("B"),
                        px: fval(&r["px"]),
                        // crossed=true 是吃单方,它的单从不进簿(回执是 filled),
                        // 只有挂单方(false)才是「我们必须持有」的那一侧。
                        crossed: r["crossed"].as_bool() == Some(true),
                    }));
                    nfills += 1;
                }
            }
        }
    }
    eprintln!("fills: {:?} 命中 {} 笔", coins, nfills);

    let no_diffs = args.get("--no-diffs").map(|s| s == "1" || s == "true").unwrap_or(false);
    let mut books: Vec<Book> = (0..nb).map(|_| {
        let mut b = Book::default();
        b.diff_on = !no_diffs;
        b.cur_ai = -1;   // 播种注入/块末阶段都不属于任何动作

        b.trust_gone = args.get("--trust-gone").map(|s| s == "1").unwrap_or(false);
        b.filled_rest = args.get("--filled-rest").map(|s| s == "1").unwrap_or(false);
        b.infer_oid = args.get("--infer-oid").map(|s| s == "1").unwrap_or(true);
        b.fresh_arbit = args.get("--fresh-arbit").map(|s| s == "1").unwrap_or(true);
        b.rule4 = args.get("--rule4").map(|s| s == "1").unwrap_or(true);
        b.sweep_clean = args.get("--sweep-clean").map(|s| s == "1").unwrap_or(true);
        b.rebind_emit = args.get("--rebind-emit").map(|s| s == "1").unwrap_or(false);
        b.ro_proof = args.get("--ro-proof").map(|s| s == "1").unwrap_or(false);
        b.ro_dedup = args.get("--ro-dedup").map(|s| s == "1").unwrap_or(false);
        b.q_order = args.get("--q-order").map(|s| s == "1").unwrap_or(false);
        b
    }).collect();
    // diffs 流式落盘:攒整窗的 Vec 在 20 币规模是 ~90GB(实测 OOM),
    // 改为每块 drain 一次写进 gz。warm_end 过滤在写入时做。
    let mut diff_ws: Vec<Option<flate2::write::GzEncoder<std::io::BufWriter<File>>>> =
        (0..nb).map(|bi| if no_diffs { None } else {
            Some(flate2::write::GzEncoder::new(
                std::io::BufWriter::with_capacity(1 << 20,
                    File::create(format!("{}/diffs_{}.ndjson.gz", out, coins[bi])).unwrap()),
                flate2::Compression::default()))
        }).collect();
    // --taker-tape 1:另落一条「吃单带」—— 每个立即成交的吃单(回执 filled 而非 resting)
    // 一行 {block, ai, side, px(限价), sz(下单量), user}。它是 C 臂(动作级结算)的块内定序原料:
    // 挂单/撤单在 diffs 里带 ai,而成交流没有块内序号 —— 只有吃单动作自己知道它在块内第几位。
    // 不写进 diffs:diffs 的逐位不变性已经被 45 个检查点的自检背书,不动它。
    let taker_tape = args.get("--taker-tape").map(|s| s == "1").unwrap_or(false);
    let mut tape_ws: Vec<Option<flate2::write::GzEncoder<std::io::BufWriter<File>>>> =
        (0..nb).map(|bi| if !taker_tape { None } else {
            Some(flate2::write::GzEncoder::new(
                std::io::BufWriter::with_capacity(1 << 20,
                    File::create(format!("{}/taker_{}.ndjson.gz", out, coins[bi])).unwrap()),
                flate2::Compression::default()))
        }).collect();
    let (mut first_blk, mut nblk, mut last_ts, mut last_blk) = (None::<u64>, 0u64, 0i64, 0u64);
    let files = sorted_gz_files(&format!("{}/actions", data));
    let nfiles = files.len();
    let (rx, worker_handles) = parse_pipeline(files, workers, std::sync::Arc::new(assets_map.clone()), mod_err);
    let mut pendbuf: HashMap<usize, Vec<BlockC>> = HashMap::default();

    // ---------- 检查点播种 ----------
    let load_seed = |raw: &[u8], label: &str| -> (Vec<Value>, u64) {
        let txt = String::from_utf8_lossy(raw).to_string();
        let mut j: Value = serde_json::from_str(&txt).unwrap();
        if j.get("data").is_some() { j = j["data"].clone(); }
        let lb = j["last_block_number"].as_u64().unwrap_or(0);
        let mut v: Vec<Value> = Vec::new();
        for k in ["bids", "asks"] {
            if let Some(arr) = j[k].as_array() { v.extend(arr.iter().cloned()); }
        }
        eprintln!("待播种[{}] {} 笔 (0x last_block={})", label, v.len(), lb);
        (v, lb)
    };
    let mut seed_orders: Vec<Option<Vec<Value>>> = vec![None; nb];
    // 注入块:该簿从这个块起开始消费 op/fill,并在消费前一刻注入种子。
    // seed-dir 逐币 = last_block+1(防双重应用)。
    let mut inject: Vec<u64> = vec![0; nb];
    let mut planted: Vec<bool> = vec![true; nb];   // 无种子的簿视同已注入
    if let Some(sd) = &seed_dir {
        for (i, c) in coins.iter().enumerate() {
            let fp_gz = format!("{}/{}_{}.json.gz", sd, c, seed_block);
            let fp_pl = format!("{}/{}.json", sd, c);
            let raw = if let Ok(f) = File::open(&fp_gz) {
                let mut r = Vec::new();
                std::io::Read::read_to_end(&mut GzDecoder::new(f), &mut r).unwrap(); r
            } else {
                std::fs::read(&fp_pl).unwrap_or_else(|_| panic!("读不到 seed {} / {}", fp_gz, fp_pl))
            };
            let (v, lb) = load_seed(&raw, c);
            seed_orders[i] = Some(v);
            inject[i] = lb + 1;
            planted[i] = false;
        }
    }
    let mut seed_any = seed_orders.iter().any(|s| s.is_some());
    // 采收:播种前只记 resting 单的 cloid/ro,不建簿
    let mut harvest: HashMap<u64, (Option<String>, bool)> = HashMap::default();
    let mut seed_done = false;
    let mut want = 0usize;
    let mut done = false;

    while want < nfiles && !done {
        let batch = match pendbuf.remove(&want) {
            Some(v) => v,
            None => match rx.recv() {
                Ok((idx, v)) => { if idx != want { pendbuf.insert(idx, v); continue; } v }
                // 分片模式下上游 chunk 数少于 JSON 文件数,读完即正常结束
                Err(_) => break,
            }
        };
        want += 1;
        for blk in batch {
            let n = blk.n;
            if n > to_block { done = true; break; }
            if first_blk.is_none() { first_blk = Some(n); }
            nblk += 1; last_blk = n;
            if let Some(t) = blk.ts { last_ts = t; }

            // ---------- 播种前采收 / 到点注入 ----------
            let min_inject = inject.iter().zip(&planted)
                .filter(|(_, p)| !**p).map(|(i, _)| *i).min();
            if seed_any && !seed_done && min_inject.is_some() {
                if n < min_inject.unwrap() {
                    {
                        for op in &blk.ops {
                            match op {
                                Op::Add(a) | Op::Meta(a) => {
                                    harvest.insert(a.oid, (a.cloid.clone(), a.ro)); }
                                Op::BatchItem { add: Some(a), .. } => {
                                    harvest.insert(a.oid, (a.cloid.clone(), a.ro)); }
                                _ => {}
                            }
                        }
                    }
                    if let Some(fs) = fills_by_block.remove(&n) {
                        for (bi, f) in fs {
                            if let Some(sp) = f.sp {
                                let mut e = sp + if f.side_b { f.sz } else { -f.sz };
                                if e.abs() < 1e-9 { e = 0.0; }
                                books[bi as usize].pos.insert(f.user.clone(), e);
                            }
                        }
                    }
                    if nblk % 20000 == 0 {
                        eprintln!("  采收中 {} 块, cloid库 {}", nblk, harvest.len());
                    }
                    continue;
                }
                // 到达注入点:逐簿注入(只注到点的)
                for bi in 0..nb {
                if planted[bi] || n < inject[bi] { continue; }
                let v = match seed_orders[bi].take() { Some(v) => v, None => continue };
                planted[bi] = true;
                let b = &mut books[bi];
                let (mut nc, mut nr) = (0u64, 0u64);
                for o in &v {
                    let oid = o["oid"].as_u64().unwrap();
                    let u = o["user_address"].as_str().unwrap_or("").to_string();
                    let (cl, ro) = harvest.get(&oid).cloned().unwrap_or((None, false));
                    if cl.is_some() { nc += 1; }
                    if ro { nr += 1; }
                    let side = if o["side"].as_str() == Some("B") { b'B' } else { b'A' };
                    let spx = fval(&o["price"]);
                    b.book.insert(oid, Entry { user: u.clone(), side,
                        px: spx, sz: fval(&o["size"]), ro, q: 0 });
                    {
                        let m = if side == b'B' { &mut b.pb } else { &mut b.pa };
                        *m.entry(tick(spx)).or_insert(0) += 1;
                    }
                    b.user_oids.entry(u.clone()).or_default().insert(oid);
                    if let Some(c) = cl {
                        b.cloid.entry((u.clone(), c.clone())).or_default().insert(oid);
                        b.oid_ck.insert(oid, (u, c));
                    }
                }
                eprintln!("播种注入[{}]: {} 笔, cloid 补挂 {} ({:.1}%), ro 标记 {}",
                          coins[bi], v.len(), nc, nc as f64 / v.len().max(1) as f64 * 100.0, nr);
                }
                if planted.iter().all(|p| *p) {
                    seed_done = true; seed_any = false; harvest.clear();
                }
            }

            // scheduleCancel 到期(先于本块动作)
            for b in books.iter_mut() {
            if !b.sched.is_empty() {
                let mut due: Vec<String> = b.sched.iter()
                    .filter(|(_, t)| last_ts >= **t).map(|(u, _)| u.clone()).collect();
                due.sort_unstable();
                for u in due {
                    let mut oids: Vec<u64> = b.user_oids.get(&u)
                        .map(|s| s.iter().cloned().collect()).unwrap_or_default();
                    oids.sort_unstable();
                    for oid in oids { b.rm(n, oid, "sched"); }
                    b.sched.remove(&u);
                }
            }
            }

            // (簿号, 发起这条 cancelByCloid 的动作 actionIndex, key)
            let mut pending_cloid: Vec<(u32, i32, (String, String))> = Vec::new();
            let mut touched: Vec<HashSet<String>> = (0..nb).map(|_| HashSet::default()).collect();

            let mut op_off = 0usize;
            for act in &blk.acts {
                let user: &str = &act.user;
                let ops_slice = &blk.ops[op_off..op_off + act.n as usize];
                let tag_slice = &blk.tags[op_off..op_off + act.n as usize];
                op_off += act.n as usize;
                for (oi, op) in ops_slice.iter().enumerate() {
                    let tag = tag_slice[oi];
                    if tag == TAG_ALL {
                        // 账户级:广播(目前只有 Sched)
                        if let Op::Sched { time } = op {
                            for b in books.iter_mut() { match time {
                                Some(t) => { b.sched.insert(user.to_string(), *t); }
                                None => { b.sched.remove(user); }
                            } }
                        }
                        continue;
                    }
                    let bi = tag as usize;
                    if !planted[bi] { continue; }     // 该簿尚未注入:本块不属于它
                    let b = &mut books[bi];
                    b.cur_ai = act.ai as i32;         // 落盘流水按动作归因,块末阶段会复位 -1
                    let touched = &mut touched[bi];
                    match op {
                        Op::Bump(k) => b.bump(k),
                        Op::Add(a) => {
                            b.add(n, user, AddInfo {
                                oid: a.oid, side: a.side, px: a.px, sz: a.sz,
                                cloid: a.cloid.clone(), ro: a.ro, certain: true }, "add");
                            b.fresh_conf.push(a.oid);
                            if a.ro { touched.insert(user.to_string()); }  // RO 单也要触发裁剪
                        }
                        Op::Cancel { oid } => {
                            if b.book.contains_key(oid) { b.rm(n, *oid, "cancel"); }
                            else {
                                match b.take_syn(user, None) {
                                    Some(o) => b.rm(n, o, "cancel_syn"),
                                    None => b.bump("rm_miss_cancel"),
                                }
                            }
                        }
                        // 交易所说这单不存在。在簿上就删,不在就算了 —— 绝不猜合成 oid。
                        Op::CancelGone { oid } => {
                            // 必须校验单主:交易所那句「Order was never placed, already
                            // canceled, or filled」不是对整个簿的断言,是对**发信人自己那部分簿**
                            // 的断言 —— 三个并列理由里的第一个 never placed,恰恰就是
                            // 「这个 oid 不是你的」时走的分支。
                            //
                            // 不校验的后果(实测反例,已逐条复核):
                            //   块 1096407341 地址 0x7717a7a2 挂买 0.002 BTC @ 62650;
                            //   块 1096407431 地址 0x1ed8d101 发一条 24 腿撤单、全部被拒,
                            //   腿里带了这个 oid → 我们把别人的活单删了;
                            //   块 1096407487/91 这张单在交易所真的成交了 0.00087+0.00113 = 0.002。
                            // 全网格 45 个窗口共 28 个互不重叠的硬反例,28/28 全是这一对地址。
                            // oid 是全交易所共用的单调计数器、可枚举,所以不校验等于
                            // 在我们的簿上开了一个不需要鉴权的写口。
                            //
                            // 同一条机制的 cloid 路径本来就按 (user, cloid) 查表 —— 是认人的。
                            // 所以这里是漏写,不是设计取舍。
                            let mine = b.book.get(oid).map_or(false, |e| e.user == user);
                            if b.trust_gone && mine { b.rm(n, *oid, "gone"); }
                            else {
                                if b.book.contains_key(oid) { b.bump("gone_not_owner"); }
                                b.bump("cancel_failed");
                            }
                        }
                        Op::CancelCloidGone { cloid } => {
                            if !b.trust_gone { b.bump("cancelcloid_failed"); }
                            else {
                                let key = (user.to_string(), cloid.clone());
                                match b.cloid.get(&key) {
                                    None => b.bump("cancelcloid_failed"),
                                    Some(oids) => {
                                        let mut live: Vec<u64> = oids.iter()
                                            .filter(|o| b.book.contains_key(o)).cloned().collect();
                                        if live.is_empty() { b.bump("cancelcloid_failed"); }
                                        else {
                                            live.sort_unstable();
                                            for o in live { b.rm(n, o, "gone"); }
                                            b.cloid.remove(&key);
                                        }
                                    }
                                }
                            }
                        }
                        Op::CancelCloid { cloid } => {
                            // 统一推迟到块末执行(见下方 pending_cloid)
                            pending_cloid.push((tag, act.ai as i32,
                                                (user.to_string(), cloid.clone())));
                        }
                        Op::Modify { target, add, trigger } => {
                            let side_new = add.as_ref().map(|a| a.side);
                            let hits: Vec<u64> = b.resolve_target(user, target)
                                .into_iter().filter(|h| b.book.contains_key(h)).collect();
                            if !hits.is_empty() {
                                for h in hits { b.rm(n, h, "modify"); }
                            } else {
                                // 目标是上一轮 modify 造的、我们只有合成 oid 的那笔
                                let s2 = b.take_syn(user, side_new)
                                    .or_else(|| b.take_syn(user, None));
                                match s2 {
                                    Some(o) => b.rm(n, o, "modify_syn"),
                                    None => b.bump("modify_target_miss"),
                                }
                            }
                            if *trigger { b.bump("trigger_skipped"); }
                            else if !syn_mod { b.bump("modify_new_dropped"); }
                            else if let Some(a) = add {
                                let (side, px, sz, ro) = (a.side, a.px, a.sz, a.ro);
                                // 认证过的用算出来的真 oid;撞号(簿上已有)一律退合成,
                                // 绝不把两笔不同的单当成同一笔。
                                let use_real = b.infer_oid && a.certain && a.oid != 0
                                               && !b.book.contains_key(&a.oid);
                                if a.certain && a.oid != 0 && b.book.contains_key(&a.oid) {
                                    b.bump("modify_oid_collision");
                                }
                                let oid = if use_real { a.oid } else { b.new_syn() };
                                b.add(n, user, AddInfo { oid, side, px, sz,
                                                          cloid: a.cloid.clone(), ro,
                                                          certain: use_real },
                                      if use_real { "add_modify" } else { "add_modsyn" });
                                b.fresh_unconf.push(oid);
                                if use_real { b.bump("modify_oid_certified"); }
                                else {
                                    b.bump("modify_oid_synthetic");
                                    // 只有合成 oid 才需要迟绑定的候选池
                                    b.pending.entry((user.to_string(), side, px.to_bits()))
                                        .or_default().push_back(oid);
                                    b.user_syn.entry(user.to_string()).or_default().push(oid);
                                    // #1 唯一性冲突:此刻该地址手上有几笔待定单
                                    let k = b.user_syn.get(user).map_or(0, |v| v.len());
                                    if k > 1 { b.bump("modify_syn_ambiguous"); }
                                    else { b.bump("modify_syn_unique"); }
                                }
                                if ro { touched.insert(user.to_string()); }
                            }
                        }
                        Op::BatchItem { target, add, trigger } => {
                            let hits = b.resolve_target(user, target);
                            if hits.is_empty() { b.bump("rm_miss_batchModify"); }
                            for h in hits { b.rm(n, h, "batchModify"); }
                            if *trigger { b.bump("trigger_skipped"); }
                            else if let Some(a) = add {
                                let ro = a.ro;
                                b.add(n, user, AddInfo { oid: a.oid, side: a.side, px: a.px,
                                                          sz: a.sz, cloid: a.cloid.clone(), ro,
                                                          certain: true }, "add_batch");
                                b.fresh_conf.push(a.oid);
                                if ro { touched.insert(user.to_string()); }
                            } else { b.bump("batchmodify_new_rejected"); }
                        }
                        Op::AddRest(a) => {
                            if b.filled_rest {
                                b.add(n, user, AddInfo { oid: a.oid, side: a.side, px: a.px,
                                                          sz: a.sz, cloid: a.cloid.clone(), ro: a.ro,
                                                          certain: true },
                                      "add_filledrest");
                                b.fresh_unconf.push(a.oid);
                                b.bump("filled_rest_added");
                                if a.ro { touched.insert(user.to_string()); }
                            } else { b.bump("filled_rest_dropped"); }
                        }
                        Op::Meta(_) => {}          // 只供播种采收,重放阶段无副作用
                        Op::Sched { time } => match time {
                            Some(t) => { b.sched.insert(user.to_string(), *t); }
                            None => { b.sched.remove(user); }
                        },
                        Op::Aggr { side, px, sz, oid } => {
                            // oid = 吃单自己的编号(回执 filled.oid)。fills 流的 crossed=true 腿
                            // 带同一个号 —— 由此把每笔逐档成交量精确挂到块内 ai 上。
                            if let Some(w) = tape_ws[bi].as_mut() {
                                writeln!(w, "{}", serde_json::json!({
                                    "block": n, "ai": act.ai, "side": (*side as char).to_string(),
                                    "px": px, "sz": sz, "oid": oid, "user": user })).unwrap();
                            }
                        }
                    }
                }
            }

            // 块末统一执行 cancelByCloid
            for (cbi, cai, key) in pending_cloid {
                let b = &mut books[cbi as usize];
                b.cur_ai = cai;
                match b.cloid.remove(&key) {
                    None => b.bump("cancelcloid_unknown"),
                    Some(oids) => {
                        let mut live: Vec<u64> =
                            oids.into_iter().filter(|o| b.book.contains_key(o)).collect();
                        if live.is_empty() { b.bump("cancelcloid_unknown"); continue; }
                        live.sort_unstable();
                        for o in live { b.rm(n, o, "cancelByCloid"); }
                    }
                }
            }

            // 动作阶段结束。从这里往后的一切(成交扣减、扫盘清理、块末仲裁)都不由
            // 单个动作直接造成 —— ai 复位 -1,流水里因此能一眼分清「动作干的」和「块末判的」。
            for b in books.iter_mut() { b.cur_ai = -1; }

            // 成交扣减 + 仓位跟踪(逐簿)
            let mut fills_grp: Vec<Vec<FillC>> = (0..nb).map(|_| Vec::new()).collect();
            if let Some(fs) = fills_by_block.remove(&n) {
                for (bi, f) in fs { fills_grp[bi as usize].push(f); }
            }
            for bi in 0..nb {
            if !planted[bi] { continue; }
            let b = &mut books[bi];
            let touched = &mut touched[bi];
            let mut sweep_buy: Option<i64> = None;    // 本块买方扫盘触及的最高价
            let mut sweep_sell: Option<i64> = None;   // 本块卖方扫盘触及的最低价
            {
                for f in fills_grp[bi].drain(..) {
                    if f.crossed {
                        let t = tick(f.px);
                        if f.side_b { sweep_buy = Some(sweep_buy.map_or(t, |x| x.max(t))); }
                        else { sweep_sell = Some(sweep_sell.map_or(t, |x| x.min(t))); }
                    }
                    if f.sp.is_some() && !f.user.is_empty() {
                        let mut end = f.sp.unwrap() + if f.side_b { f.sz } else { -f.sz };
                        if end.abs() < 1e-9 { end = 0.0; }
                        b.pos.insert(f.user.clone(), end);
                        touched.insert(f.user.clone());
                    }
                    if !b.book.contains_key(&f.oid) {
                        // 延迟绑定:这笔成交的 oid 没见过,看是不是某条 modify 造的合成单
                        let key = (f.user.clone(),
                                   if f.side_b { b'B' } else { b'A' }, f.px.to_bits());
                        loop {
                            let syn = match b.pending.get_mut(&key) {
                                Some(q) => q.pop_front(), None => None };
                            match syn {
                                None => { b.pending.remove(&key); break; }
                                Some(s0) => if b.book.contains_key(&s0) && b.rebind(n, s0, f.oid) { break },
                            }
                        }
                    }
                    // 先在一次可变借用里改完 Entry 并把需要的字段拷出来,再动价位索引
                    // 成交重放判定:成交发生前簿上必须有这笔单,且余量 >= 成交量
                    let short = b.book.get(&f.oid).map_or(false, |e| e.sz + 1e-9 < f.sz);
                    if short { b.bump("fill_size_short"); }
                    let hit = match b.book.get_mut(&f.oid) {
                        None => None,
                        Some(e) => {
                            b.filled_blk.insert(f.oid);
                            e.sz -= f.sz;
                            Some((e.side, e.px, e.sz, e.user.clone()))
                        }
                    };
                    match hit {
                        None => {
                            b.bump("fill_oid_not_in_book");
                            if f.crossed { b.bump("fill_miss_taker"); }
                            else { b.bump("fill_miss_born"); }
                        }
                        Some((side, px, szl, ul)) => {
                            if !f.crossed { b.bump("fill_hit_born"); }
                            if szl <= 1e-8 {
                                b.rm(n, f.oid, "fill");
                            } else {
                                // 与 add/rm 同款守卫:--no-diffs 时不落 diff,计数照 bump
                                if b.diff_on {
                                    b.diffs.push(Diff { block: n, oid: f.oid, typ: "update", why: "fill_partial",
                                                        side, px, sz: szl, user: ul, ai: -1 });
                                }
                                b.bump("update_fill");
                            }
                        }
                    }
                }
            }

            // ---------- 扫盘清理:成交流证明已死的对侧滞留单 ----------
            // 买方扫盘吃到了 P,说明真实卖侧 P 以下已被吃穿。我们簿上任何
            // 低于 P、又没在本块被成交打中的老卖单,被交易所的成交流证明是死的。
            // 与已退役的 --uncross 启发式的本质区别:证据是交易所的成交,不是年龄;
            // 且本块新增的单一律豁免(fills 无块内序号,后挂的确认单可合法在扫内)。
            // 实测抓到的反例正是它救回来的:75.9 BTC 真单曾因陈年灰尘卖单(63440)
            // 污染 touch 而被穿价检查错杀。
            // 适配器①:成交流扫价。bound = 成交价(strict:成交价那一档可以合法留着,
            // 部分成交),qb = BeforeBlock —— fills 没有块内序号,只能整块豁免。
            // 注:这里的 BeforeBlock 比原实现多了一个「本块被打中还有余量也豁免」的条件,
            // 与 rule4 的粗口径统一。原实现只查 fresh_all,是三份实现口径不一的遗留。
            if b.sweep_clean {
                if let Some(sb) = sweep_buy {
                    b.arbitrate(n, b'A', sb, true, QBound::BeforeBlock, None,
                                "swept", "swept", None);
                }
                if let Some(ss) = sweep_sell {
                    b.arbitrate(n, b'B', ss, true, QBound::BeforeBlock, None,
                                "swept", "swept", None);
                }
            }
            // ---------- 未确认入口的穿价剩余 ----------
            // filled-rest / modify 插入的单,交易所从没说过它 resting。同块成交
            // 已经扣完之后,如果它的限价仍穿过我们对侧最优价,那它物理上不可能
            // 挂在真实簿上(撮合引擎会当场吃掉或按 STP/保证金取消,且取消无记录
            // —— 实测:order Gtc 卖 0.41162@6400,交易所只成交 0.05966,剩余被
            // 静默取消;我们却让它挂着,以 6400 定义了我们的最优卖价,偏离 3958bp)。
            if !b.fresh_arbit { b.fresh_unconf.clear(); }
            if !b.fresh_unconf.is_empty() {
                let fresh = std::mem::take(&mut b.fresh_unconf);
                let fset: HashSet<u64> = fresh.iter().cloned().collect();
                for oid in fresh {
                    let (side, px) = match b.book.get(&oid) {
                        Some(e) => (e.side, e.px), None => continue };
                    let crossing = if side == b'B' {
                        b.pa.keys().next().map_or(false, |&a| tick(px) >= a)
                    } else {
                        b.pb.keys().next_back().map_or(false, |&x| tick(px) <= x)
                    };
                    if !crossing { continue; }
                    // 与它穿价的对侧单,分两类:
                    //  活证据 = 本块被成交打中且扣完仍有量(STP 案例:对手价位还在,
                    //           我们的剩余量确实被交易所取消了) 或 本块刚确认 resting
                    //  无证据 = 陈年滞留(鲸鱼案例:75.9 BTC 真单 vs 63440 死灰尘,
                    //           真单的 resting 本身就证明这些灰尘已死)
                    let t0 = tick(px);
                    let contra: Vec<(u64, bool)> = b.book.iter()
                        .filter(|(o, e)| **o != oid && e.side != side && !fset.contains(o)
                                && if side == b'B' { tick(e.px) <= t0 } else { tick(e.px) >= t0 })
                        .map(|(o, e)| (*o, (b.filled_blk.contains(o) && e.sz > 1e-9)
                                           || b.fresh_all.contains(o)))
                        .collect();
                    if contra.iter().any(|(_, live)| *live) {
                        b.rm(n, oid, "aggr_drop");          // 有活证据顶着:剩余量被取消
                    } else {
                        for (o, _) in contra { b.rm(n, o, "stale_cross_fa"); }  // 只有死灰尘:清灰尘留真单
                    }
                }
            }
            // ---------- 规则④:确认 resting 回执 = 内侧死亡证明 ----------
            if !b.rule4 { b.fresh_conf.clear(); }
            if !b.fresh_conf.is_empty() {
                let fresh = std::mem::take(&mut b.fresh_conf);
                for oid in fresh {
                    let (side, px, q0) = match b.book.get(&oid) {
                        Some(e) => (e.side, e.px, e.q), None => continue };
                    let crossing = if side == b'B' {
                        b.pa.keys().next().map_or(false, |&a| tick(px) >= a)
                    } else {
                        b.pb.keys().next_back().map_or(false, |&x| tick(px) <= x)
                    };
                    if !crossing { continue; }
                    let t0 = tick(px);
                    // 块内定序(--q-order):原口径把「本块新增」和「本块被打中」整体豁免,
                    // 因为 fills 没有块内序号。但 resting 回执是**动作**,而动作是按
                    // actionIndex 排过序的(见 parse 里的 acts.sort_by_key),入簿序号 q
                    // 随之严格递增 —— q 就是块内的钟。
                    // 于是豁免的正确口径不是「同块」,而是「比我晚」:q > q0 的才豁免。
                    // 实测现场 ACE 块 1103044787:同一地址 ai204 挂买 0.1523(Alo,resting),
                    // ai212 自己发 Ioc 卖 0.1523 触发 STP 静默撤掉这张买单(1187.26 只成交
                    // 551.8 = 撞到自己),ai215 另一地址的 Gtc 卖 0.1523 拿到 resting。
                    // 旧口径:买单同块新增 + 同块被打中,双重豁免 → 留簿 → 锁价 2 块。
                    // 新口径:q(买)< q(卖) → 后者的 resting 回执即前者的死亡证明。
                    // 顺带把「谁的功劳」分开记账:旧口径本来就会杀的记 stale_cross_r4,
                    // 只有靠块内定序才杀得到的(旧口径会豁免它)记 stale_cross_q ——
                    // 这样落盘流水里 q-order 的边际杀伤是可以直接数出来的。
                    // 适配器③:resting 回执。bound = 该单价格(非 strict:同价锁定也算穿价),
                    // qb = Exact(q0) —— 回执是动作,有块内序号,这是最精确的一类证据。
                    let kill_side = if side == b'B' { b'A' } else { b'B' };
                    let qb = if b.q_order { QBound::Exact(q0) } else { QBound::BeforeBlock };
                    b.arbitrate(n, kill_side, t0, false, qb, Some(oid),
                                "stale_cross_r4", "stale_cross_q", Some("rest_assert_kill"));
                }
            }
            b.fresh_all.clear();
            b.filled_blk.clear();
            let mut tu: Vec<String> = touched.drain().collect();
            tu.sort_unstable();
            for u in tu { b.ro_trim(n, &u); }

            // ---------- 块末自证:簿不该穿价 ----------
            // 不用任何外部真值。撮合引擎的定义性质:静息态 最优买 < 最优卖。
            // 这里只**记数不改簿**(改簿是 --uncross 的老启发式,已退役),
            // 让「还剩几块穿价」成为引擎自己报出来的验收指标。
            //
            // 位置必须在**块末处理的最后一行**。第一版放在 ro_trim 之前,
            // 于是 HYPE 报出 1 块假阳性:块 1101633245 的买单 512436376972 正是被
            // 同块的 ro_trim(remove_ro_flat)清掉的,探针在它清掉之前就拍了照。
            // 独立的 python 扫描器(重放 diffs)当时报 0 —— 两边不一致时,
            // 错的是探针的站位,不是簿。
            if let (Some(&x), Some(&y)) = (b.pb.keys().next_back(), b.pa.keys().next()) {
                if x >= y {
                    b.xblocks += 1;
                    if b.xsites.len() < 2000 {
                        // 现场取证:各挑一张钉住盘口的单(同价位取最老的,即队首)
                        let pick = |bk: &HashMap<u64, Entry>, s: u8, t: i64| -> (u64, f64) {
                            bk.iter().filter(|(_, e)| e.side == s && tick(e.px) == t)
                              .min_by_key(|(o, e)| (e.q, **o))
                              .map(|(o, e)| (*o, e.px)).unwrap_or((0, 0.0))
                        };
                        let (bo, bpx) = pick(&b.book, b'B', x);
                        let (ao, apx) = pick(&b.book, b'A', y);
                        b.xsites.push((n, bpx, apx, bo, ao));
                    }
                }
            }

            }   // ← 逐簿块末处理结束

            // diffs 流式写盘(warm_end 之前的丢弃,与旧尾部写法同口径)
            if !no_diffs {
                let we = first_blk.unwrap_or(0) + warmup;
                for bi in 0..nb {
                    if books[bi].diffs.is_empty() { continue; }
                    let w = diff_ws[bi].as_mut().unwrap();
                    for d in books[bi].diffs.drain(..) {
                        if d.block < we { continue; }
                        writeln!(w, "{}", serde_json::json!({
                            "block": d.block, "oid": d.oid, "type": d.typ, "why": d.why,
                            "side": (d.side as char).to_string(), "px": d.px, "sz": d.sz,
                            "user": d.user, "ai": d.ai })).unwrap();
                    }
                }
            }

            // 快照:固定网格(--snap-at),每行一个毫秒时间戳
            if last_ts > 0 {
                // (要落盘的编号, 对账时要查的时间戳) —— 后者必须是网格上的标称值,
                // 不是本块的 last_ts,否则网格又漂了。两者最多差一个块(约 71ms)。
                let mut due: Option<(i64, i64)> = None;
                if !snap_at.is_empty() {
                    // 一次可能跨过多个网格点(块间隔大时),只落最后一个,中间的补不回来
                    while snap_i < snap_at.len() && snap_at[snap_i] <= last_ts {
                        due = Some((snap_i as i64, snap_at[snap_i]));
                        snap_i += 1;
                    }
                }
                // 热身区不出快照:此时簿子还缺窗口之前挂的单,对账没有意义
                if let Some((k, grid_ts)) = due.filter(|_| n >= emit_from) {
                    for (bi, b) in books.iter().enumerate() {
                    let cn = &coins[bi];
                    let mut orders: Vec<Value> = Vec::with_capacity(b.book.len());
                    for (o, e) in &b.book {
                        orders.push(serde_json::json!({"oid": o, "user": e.user,
                            "side": (e.side as char).to_string(), "px": e.px,
                            "sz": e.sz, "ro": e.ro, "q": e.q}));
                    }
                    let fp = format!("{}/hsnap_{}_{:03}_{}.json.gz", out, cn, k, n);
                    let f = File::create(&fp).unwrap();
                    let mut gz = flate2::write::GzEncoder::new(f, flate2::Compression::new(6));
                    let doc = serde_json::json!({"coin": cn, "last_block": n,
                        "last_ts": grid_ts, "actual_ts": last_ts, "hour": k, "orders": orders});
                    gz.write_all(serde_json::to_string(&doc).unwrap().as_bytes()).unwrap();
                    gz.finish().unwrap();
                    }
                    eprintln!("  [快照 h{}] block {} ts {} (实际 {:+}ms) 簿共 {} 笔",
                              k, n, grid_ts, last_ts - grid_ts,
                              books.iter().map(|b| b.book.len()).sum::<usize>());
                }
            }

            if nblk % 50000 == 0 {
                eprintln!("  {} 万块, 簿共 {} 笔", nblk / 10000,
                          books.iter().map(|b| b.book.len()).sum::<usize>());
            }
        }
    }

    // 回收工人线程:先关接收端(提前结束 --to-block 时工人可能还阻塞在 send 上,
    // 关掉让它们退出),再逐个 join;任何一个 panic 都说明输入没读全,不能装作跑完。
    drop(rx);
    let mut worker_dead = false;
    for h in worker_handles {
        if h.join().is_err() { worker_dead = true; }
    }
    if worker_dead {
        eprintln!("工人线程异常退出,数据可能截断");
        std::process::exit(3);
    }

    eprintln!("\n重放完成: block {:?} → {} ({} 块), 簿共 {} 笔",
              first_blk, last_blk, nblk,
              books.iter().map(|b| b.book.len()).sum::<usize>());
    for (bi, b) in books.iter().enumerate() {
        eprintln!("== {} 簿 {} 笔 == 自检:块末穿价/锁定 {} 块 ({:.6}%)",
                  coins[bi], b.book.len(), b.xblocks,
                  b.xblocks as f64 / nblk.max(1) as f64 * 100.0);
        let xf = format!("{}/xcross_{}.json", out, coins[bi]);
        let sites: Vec<Value> = b.xsites.iter().map(|(n, bp, ap, bo, ao)| serde_json::json!(
            {"block": n, "bid": bp, "ask": ap, "bid_oid": bo, "ask_oid": ao})).collect();
        std::fs::write(&xf, serde_json::to_string(&serde_json::json!(
            {"coin": coins[bi], "blocks": nblk, "xblocks": b.xblocks,
             "truncated": b.xblocks as usize > b.xsites.len(), "sites": sites})).unwrap()).ok();
        let mut stats: Vec<(&&'static str, &u64)> = b.st.iter().collect();
        stats.sort_by(|a, c| c.1.cmp(a.1).then(a.0.cmp(c.0)));
        for (k, v) in &stats { eprintln!("  {:32} {:>10}", k, v); }
    }

    // 输出与 Python 版同构的产物。
    // 5 天的 diff 是 2.2GB,单线程 gzip 要写十几分钟;不需要 diffs 时用 --no-diffs 跳过。
    let warm_end = first_blk.unwrap_or(0) + warmup;
    // 取值式:argv 是严格成对步进的,裸 flag 会错位后面的参数。用法 --no-diffs 1
    for (bi, b) in books.iter().enumerate() {
    let coin = &coins[bi];
    if no_diffs {
        if bi == 0 { eprintln!("跳过 diffs 输出 (--no-diffs)"); }
    } else if let Some(w) = diff_ws[bi].take() {
        let _ = warm_end;                    // 过滤已在流式写入时做
        w.finish().unwrap();
        eprintln!("diffs_{} 流式写盘完成", coin);
    }
    if let Some(w) = tape_ws[bi].take() {
        w.finish().unwrap();
        eprintln!("taker_{} 吃单带写盘完成", coin);
    }

    let orders: Vec<Value> = b.book.iter().map(|(o, e)| serde_json::json!({
        "oid": o, "user": e.user, "side": (e.side as char).to_string(),
        "px": e.px, "sz": e.sz, "ro": e.ro })).collect();
    let snap = serde_json::json!({
        "coin": coin, "last_block": last_blk, "last_ts": last_ts, "orders": orders });
    let mut sw = flate2::write::GzEncoder::new(
        File::create(format!("{}/snapshot_{}.json.gz", out, coin)).unwrap(),
        flate2::Compression::default());
    sw.write_all(snap.to_string().as_bytes()).unwrap();
    sw.finish().unwrap();
    eprintln!("已写出 {}/snapshot_{}.json.gz", out, coin);
    }
}
