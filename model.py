
from interactive import *
from independent import *

"""CrossAttention(num_head=4)+multiscale+DNN"""


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units):
        super(DNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(0.5)

        self.hidden_units = [inputs_dim] + list(self.hidden_units)
        # 创建一系列的线性变换层（全连接层）
        self.linear = nn.ModuleList([
            nn.Linear(self.hidden_units[i], self.hidden_units[i + 1]) for i in range(len(self.hidden_units) - 1)
        ])

        self.activation = nn.Softmax()

    def forward(self, X):
        inputs = X
        for i in range(len(self.linear)):
            fc = self.linear[i](inputs)
            gate_fc = self.activation(fc)
            fc = fc * gate_fc
            fc = self.dropout(fc)
            inputs = fc

        return inputs


class DBENet_NPI(nn.Module):
    def __init__(self, protein_input_dim, rna_input_dim, protein_out_dim=128, rna_out_dim=128,
                 dnn_hidden_units=(512, 256, 128)):
        super(DBENet_NPI, self).__init__()
        self.interactive_attention1 = CrossTransformerEncoder(protein_input_dim, rna_input_dim)
        self.interactive_attention2 = CrossTransformerEncoder(512, 512)
        self.interactive_attention3 = CrossTransformerEncoder(512, 512)
        self.interactive_attention4 = CrossTransformerEncoder(512, 512)
        self.hmcn_pro = HMCN(protein_input_dim, protein_out_dim)
        self.hmcn_rna = HMCN(rna_input_dim, rna_out_dim)
        self.linear = nn.Linear(2048, 1024)
        self.dnn = DNN(protein_out_dim * 4 + rna_out_dim * 4, dnn_hidden_units)
        self.residual_connection = nn.Linear(protein_out_dim * 4 + rna_out_dim * 4, dnn_hidden_units[-1])
        self.linear_out = nn.Linear(dnn_hidden_units[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # multi-perspective information
        protein_feature, rna_feature = x[:, :829], x[:, 829:]

        # interactive feature encoder (MCANet)
        interactive_pro1, interactive_rna1 = self.interactive_attention1(protein_feature, rna_feature)
        interactive_pro2, interactive_rna2 = self.interactive_attention2(interactive_pro1, interactive_rna1)
        interactive_pro3, interactive_rna3 = self.interactive_attention3(interactive_pro2, interactive_rna2)
        interactive_pro4, interactive_rna4 = self.interactive_attention4(interactive_pro3, interactive_rna3)
        interactive_feature = torch.cat([interactive_rna4, interactive_pro4], dim=1)

        # independent feature encoder (HMCNet)
        independent_pro = protein_feature.unsqueeze(2)
        independent_rna = rna_feature.unsqueeze(2)
        independent_pro = self.hmcn_pro(independent_pro)
        independent_rna = self.hmcn_rna(independent_rna)
        independent_pro = independent_pro.squeeze(2)
        independent_rna = independent_rna.squeeze(2)
        independent_feature = torch.cat([independent_rna, independent_pro], dim=1)

        # concatenate
        all_features = torch.cat([interactive_feature, independent_feature], dim=1)

        # ncRPI prediction module
        concatenated = self.linear(all_features)
        dnn_output = self.dnn(concatenated)
        residual_output = self.residual_connection(concatenated)
        final_input = torch.add(dnn_output, residual_output)
        prediction = self.linear_out(final_input)
        return self.sigmoid(prediction)
