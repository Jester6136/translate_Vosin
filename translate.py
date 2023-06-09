from src.main import Seq2SeqTranslator
model_names = {
    'envi': 'model/envi',
    'zhvi': 'model/zhvi',
}
translator = Seq2SeqTranslator(model_names, device='cpu',batch_size=1)

def translate(language:str,query:str):
    return translator.translate(language, query)

query = """中国日报网4月4日电  英国《卫报》网站3月28日刊发评论文章称，美国是世界上最富裕的国家，但在这里为什么就喝不到一杯干净的饮用水？

文章提到，美国宾夕法尼亚州一家工厂日前发生泄漏事故，多达8000加仑（约合30283升）用于乳胶表面处理的化学物质流入在特拉华河，而这些物质是无法通过把水烧开或者过滤来去除的。不仅如此，全美各地的自来水都掺有令人不安的化学物质。

文章作者阿瓦·玛哈达维（Arwa Mahdawi）说，当她在社交媒体上看到一条说这起事故可能会影响到费城饮用水的消息时，她恰好在喝茶，然后立刻就把嘴里用特拉华河力的水沏的茶吐了出来。

玛哈达维跟家人说“最好买些瓶装水”，但最后她也只买到了两瓶。她表示，在那时，费城大约150万居民都接到了紧急电话警报，告诉人们为了“谨慎起见”，从一小时后开始应该使用瓶装水。

当费城的瓶装水都被抢购一空的时候，人们的不安情绪开始加剧，当地也发布了新的消息。消息大概的意思是，“你不需要把所有的瓶装水都买走，因为在周一晚11点59分之前，自来水都是没问题的”。

玛哈达维不仅质疑，“那之后会怎么一样呢”？当地市政府官员曾表示，“摄入受污染水的人不会出现任何短期症状”，直接回避了长期症状。

与此同时，涉事企业的一位高管也对当地媒体表示，泄露到饮用水供应系统中的化学物质并不是什么大问题。“就跟油漆里的材料一样。”他说，“就是你房子用的常见丙烯酸涂料，这就是泄露到水里的物质。”

对此，玛哈达维讽刺说：“这可不是太让人放心，因为我们当中没多少人会到处喝油漆。也许这就是整天与有毒化学品打交道的企业才能想出的声明。”

她还表示，甚至人们抢购的瓶装水也不能保证百分之百安全。一些费城人发现，他们因此次事故而购买的瓶装水正在被召回，因为它们可能受到之前俄亥俄州东巴勒斯坦镇“毒列车”脱轨事故所泄露的化学物质的污染。

玛哈达维指出，事故的发生在所难免，但不可避免事故与因企业贪得无厌、基础设施老化、监管缺失而发生的公共卫生灾难还是有区别的。“美国是世界上最富裕的国家，干净、清洁的水不应成为奢侈品。然而，它正在成为一种（奢侈品）。”她写道。

玛哈达维还提到，美国最为人熟知的污水案例是密歇根州的弗林特，当地的自来水多年来一直铅含量超标。不过，全美各地的自来水痘掺杂着令人不安的“永久性化学物质”。

英国《卫报》和美国《消费者报告》杂志（Consumer Reports）2021年进行的一项调查发现，美国数百万人的饮用水中含有潜在的有毒化学物质，其中贫困地区的比例尤其高。

玛哈达维最后指出，费城的这一事故提醒着人们，清洁的饮用水可能不再是美国人理所当然的东西。"""


print(translate(language='zhvi',query=query))