// GENERATED FILE FROM MODEL ../models/jana_nn.pth
#include "nn.h"
#include "nn_utils.h"


struct NeuralNetwork
{
	float linear_relu_0_weight[12][15];
	float linear_relu_0_bias[15];
	float linear_relu_0_activation[15];
	float linear_relu_2_weight[15][10];
	float linear_relu_2_bias[10];
	float linear_relu_2_activation[10];
	float linear_relu_4_weight[10][6];
	float linear_relu_4_bias[6];
	float linear_relu_4_activation[6];
};

struct NeuralNetwork jana_nn = {
	.linear_relu_0_weight = { { 1.0249123722314835e-01, -1.7305153608322144e-01, 3.5661584876955382e-01, 1.9382970631880636e-01, -1.0475307068644807e-01, -7.7429637556458430e-02, -4.7281273519688360e-02, -2.1193567673551506e-02, -3.1886254245734797e-04, 2.4360150098800659e-01, -1.6239964962005615e-01, 5.2622906476765749e-02, -1.2707976996898651e-01, -4.6585559612893651e-01, -1.8438161909580231e-01 }, { -2.8674656152725220e-01, 1.7645236849784851e-01, 6.4036883073153250e-02, 6.0612753773306043e-02, 2.7576522805057091e-01, -3.7407573288975496e-02, -3.4171908496997472e-01, 3.3473573521237088e-02, -5.2851158005480396e-02, 2.7937561273574829e-01, -1.4822417497634888e-01, -5.5557131872901337e-02, 2.3634196817874908e-01, 3.6449785499337570e-02, 6.7111416719853878e-03 }, { -9.5168717205524445e-02, -2.4011223018169403e-01, 2.5610235362986494e-01, 1.9667183234160646e-01, 2.3896778283522793e-01, -2.9376118040528042e-02, 3.6065096746452568e-01, -2.7043729048766559e-02, 6.4646556649295847e-02, -3.7835691124200821e-02, -1.4337421953678131e-01, -1.5494370157269288e-01, -2.6567453145980835e-01, 2.0294497982862128e-01, -6.6378422081470490e-02 }, { -2.3611140251159668e-01, 9.9011078476905823e-02, -5.8897017806908993e-04, -1.4412917718841653e-01, 7.1111042914228109e-02, -5.6377031865896834e-02, 2.1965015524389558e-01, -2.2619659272200079e-01, -4.4869342815063593e-02, 6.5555512905120850e-02, 2.5390106439590454e-01, 7.5555761620022324e-02, -3.9945468306541443e-02, 2.5842497838252616e-02, -2.0734009146690369e-01 }, { -2.9535230249166489e-02, -9.1559477150440216e-02, -2.4738037494476769e-02, 7.5404062648973091e-02, -2.0276141797762423e-02, -1.0563064289538127e-02, -1.2534560382234611e-01, -1.7859693677180022e-01, -1.8002034207712233e-02, -5.8766353875398636e-02, -2.2443903982639313e-01, -9.0994818198422103e-02, -1.0170789808034897e-01, -1.1736755065650918e-01, 2.5057378411293030e-01 }, { 1.1426997184753418e-01, -1.0635162889957428e-01, 4.4184381019235142e-01, -1.1225799829792579e-01, -4.1039273387105812e-01, -1.0377298372665034e+00, 4.4079835032783524e-02, 9.6541667108227855e-02, -1.4145573104769382e-01, 1.9439339637756348e-01, 4.1152600198984146e-02, -4.6669843055367921e-02, -4.8659984022378922e-02, -4.8397922754320330e-03, -1.4087443053722382e-01 }, { -1.5323883295059204e-01, 1.6695028543472290e-01, -2.4823536807000832e-01, -6.3153671290687685e-01, -4.2472693446240706e-01, 9.3167860960425911e-02, -3.6156390617947276e-01, -2.5280867280703445e-01, -2.0482022232305727e-01, -1.3734251260757446e-01, 3.3657431602478027e-02, 6.3058013633668164e-01, -1.8700604140758514e-01, -2.8663566638423127e-01, -1.3294838368892670e-01 }, { 1.0930410772562027e-01, 4.0325865149497986e-02, 1.6260913601927049e+00, 1.3598512472665871e+00, 6.4610696225626219e-01, 7.2440692008418162e-01, 1.6201109523181951e+00, 2.2330797494644214e-01, 3.6907200435343133e+00, 6.1862953007221222e-02, 5.5575158447027206e-02, -1.7476101912384523e+00, -6.9800704717636108e-02, 5.1783444229242193e-01, 1.9281814992427826e-01 }, { 1.3640534877777100e-01, -1.0859362035989761e-01, 1.5552216199729707e+00, 1.4702409601043458e+00, 3.5314927613896946e-01, -7.1286751079968702e-01, 1.5416051662038626e+00, -1.7240621877084231e-01, 3.5481301633731972e+00, -2.5070226192474365e-01, -2.4171368777751923e-01, -1.7371837277296713e+00, -1.6437686979770660e-01, 9.3981717949725693e-02, -2.2140648961067200e-01 }, { -2.4922558665275574e-01, -3.2567199319601059e-02, -4.6462719756489329e-01, -6.5613430473294188e-01, -4.7851579158929586e-01, 1.4802837194435725e-01, -7.3278167492807478e-01, 2.2232634129485118e-01, -2.9720312132300891e-01, 1.7613591253757477e-01, -8.0244010314345360e-03, 9.4618232242104994e-01, -2.0339384675025940e-02, -6.3458707231937561e-01, -2.1047944203019142e-02 }, { 1.2176603078842163e-01, -1.8259821459650993e-02, 3.2129529522902511e+00, 3.9772945839571072e+00, -9.3232807203696699e-01, 2.3817022705993036e+00, -1.2282494185453059e+00, 8.3144463334567043e-02, 3.1300683319306088e+00, -2.4236959218978882e-01, -1.4869952201843262e-01, 1.5907943927664105e+00, 1.1676093935966492e-01, -4.1873878582983917e+00, -4.6742364764213562e-02 }, { -2.8643953800201416e-01, 2.8426715731620789e-01, -2.5938827499607325e-02, 6.9933573593800502e-01, 4.6897179641710061e+00, -1.2606583169630400e-01, -1.8620957826348374e+00, 1.2794680744006093e-01, 3.0413975527424415e+00, -2.5327998399734497e-01, 1.9388639926910400e-01, 8.4170550772439567e-01, 7.3751114308834076e-02, 4.0510830896868039e-01, -2.0001816749572754e-01 } },
	.linear_relu_0_bias = { 0.03519224002957344, 0.27732986211776733, -0.558159045572048, -0.2512148626281709, -0.06064718945262183, 0.4010112511286343, -0.28629286452860414, -0.09816784452636734, -0.2401853072702477, 0.025662142783403397, 0.20059026777744293, 0.34114784834681344, 0.20934784412384033, -0.2820372025433657, -0.0737040713429451 },
	.linear_relu_0_activation = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
	.linear_relu_2_weight = { { 0.080594502389431, -0.08265360444784164, -0.049681153148412704, 0.007075269240885973, -0.20346176624298096, -0.1908859759569168, 0.048705220222473145, 0.0034766909666359425, -0.020203644409775734, 0.18343177437782288 }, { 0.03351081907749176, 0.02887915074825287, 0.07461939752101898, -0.17893508076667786, 0.2532367408275604, 0.20486211776733398, 0.23135437071323395, 0.1005694568157196, 0.1484406739473343, 0.18304845690727234 }, { -0.2406422346830368, -0.16328535974025726, -0.2030797302722931, 0.10517623544999594, -0.10235052669874714, 0.411540693176833, 0.10204309969020811, 0.2042108464885963, 0.028164707737838596, -0.24850687384605408 }, { 0.13438637554645538, -0.062116555869579315, -0.1931796818971634, 0.045396236233233085, 0.060198673218028735, 0.10642225041172391, -0.012524061663946576, 0.3094121532989621, -0.1735982688767367, 0.07782147079706192 }, { -0.09844264388084412, 0.12124723196029663, -0.010016300715506077, -0.01703163985106454, -0.30357115415310754, 0.010714140059248868, -0.09699901137797319, 0.25047678992073846, -0.2448338517425766, 0.06319218128919601 }, { -0.23834028840065002, 0.25394949316978455, 0.08347739279270172, -0.27982946332693187, 0.1195221126739742, 0.14481678406820372, -0.21512077885560102, -0.006746813697162747, -0.12385695527622317, -0.25756245851516724 }, { 0.04394492506980896, -0.056042857468128204, 0.13736902177333832, -0.29393121872433287, 0.33578134602779103, 0.12173436480778273, -0.25852131957086044, 0.3669210017263844, -0.2747530944526746, -0.1430806666612625 }, { 0.23310486972332, 0.004129343666136265, -0.01685330457985401, -0.13923324809633253, -0.12039617449045181, -0.27582060090818894, 0.1599324681317445, 0.16968725345968835, -0.07116098701953888, -0.13638213276863098 }, { -0.0964989960193634, -0.07974166423082352, 0.07408447563648224, -0.14697435360820763, 0.023008107255434826, 0.05240730178922875, 0.15544855510738217, 0.18307950651419175, -0.03148049346744801, 0.23528945446014404 }, { -0.042322464287281036, -0.05866082385182381, -0.20792552828788757, 0.248063862323761, 0.20922714471817017, 0.07456488162279129, -0.14134886860847473, -0.23112083971500397, 0.10243270546197891, -0.15039457380771637 }, { -0.17059326171875, 0.037465304136276245, -0.1740557849407196, -0.24340058863162994, -0.24677297472953796, 0.136837899684906, 0.16181334853172302, -0.06452047824859619, 0.19951599836349487, 0.10474254190921783 }, { -0.07422710955142975, -0.03315340727567673, -0.212086483836174, -0.17919952748902676, -0.26877835132162076, -0.03437804436091844, -0.15382765102945958, -0.7283930765981316, -0.04396910795577005, 0.1796928197145462 }, { 0.05929528921842575, 0.24396982789039612, 0.14904217422008514, 0.04666043445467949, -0.24318069219589233, -0.05757701024413109, 0.2150297909975052, -0.1738249659538269, 0.19809292256832123, 0.09948869049549103 }, { -0.025444259867072105, -0.046887002885341644, -0.24434810876846313, -0.15944324984694244, -0.020282761622241546, -0.1632207198342389, 0.004991159723041913, 0.2444214105289526, -0.13844382762908936, 0.09222852438688278 }, { -0.042003341019153595, 0.05817001312971115, 0.03997188061475754, 0.22048084437847137, 0.11563744395971298, -0.255465567111969, 0.2269771248102188, -0.1094650998711586, -0.11402150988578796, -0.16518591344356537 } },
	.linear_relu_2_bias = { -0.1778474748134613, -0.2545165717601776, -0.17402008175849915, 0.12836909212978603, 0.25913471015512707, -0.3020451636934184, 0.19612139624309302, -0.45591822685050676, 0.09146003264293583, -0.1280401349067688 },
	.linear_relu_2_activation = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
	.linear_relu_4_weight = { { -0.3086908757686615, -0.0268440879881382, -0.2686311602592468, 0.12845686078071594, 0.13014328479766846, 0.18164920806884766 }, { -0.1186172291636467, 0.14360544085502625, 0.15099561214447021, -0.06177537515759468, -0.0077873170375823975, 0.0784555971622467 }, { 0.18970869481563568, 0.12110385298728943, -0.018243640661239624, 0.20992615818977356, -0.020116591826081276, -0.11731482297182083 }, { 0.09250055880009485, -0.1328744896611299, -0.07333915457524136, 0.058245584190953374, 0.1028589135151572, -0.212775024083045 }, { -0.1753793948841576, -1.2889563774290738, 0.1298070941454537, -0.10530620406864362, 0.025709143023287626, 0.003560469560126456 }, { 1.381532825705032, -0.05723805874240607, 0.32458131792711353, 0.00441391852065872, -0.07884969135810878, 0.03541692622003226 }, { 0.1495551405726172, 0.25421757854594124, -0.07823403414340671, -0.020315446107733972, -0.07747216565495456, 0.26195494328673885 }, { -0.3862648150337382, 0.2055243583438043, 0.9629681545265449, 0.005146978846469455, 0.006351306409550228, -0.007364479140235357 }, { 0.08635911799769258, -0.006849793770776021, 0.144346761665451, -0.16116135872812085, -0.24332100445199756, 0.07807365455920241 }, { -0.25094616413116455, -0.1844460368156433, 0.2271433025598526, 0.2180703580379486, 0.2859537899494171, -0.1902109682559967 } },
	.linear_relu_4_bias = { -0.213375179369907, 0.3123180533699842, -1.0697199606450853, 0.039997060950371956, 0.1897433545008339, 0.19910695686449395 },
	.linear_relu_4_activation = { 0., 0., 0., 0., 0., 0. },
};

const float* nn_forward(float input[INPUT_SIZE]) {
	layer(12, 15, input, jana_nn.linear_relu_0_weight, jana_nn.linear_relu_0_bias, jana_nn.linear_relu_0_activation, 1);
	layer(15, 10, jana_nn.linear_relu_0_activation, jana_nn.linear_relu_2_weight, jana_nn.linear_relu_2_bias, jana_nn.linear_relu_2_activation, 1);
	layer(10, 6, jana_nn.linear_relu_2_activation, jana_nn.linear_relu_4_weight, jana_nn.linear_relu_4_bias, jana_nn.linear_relu_4_activation, 0);
	return jana_nn.linear_relu_4_activation;
};