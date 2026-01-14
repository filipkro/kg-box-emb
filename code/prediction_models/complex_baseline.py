# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, pickle
# %%
class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()

        self.emb_dim = embedding_dim

        # Real and imaginary parts
        self.ent_re = nn.Embedding(num_entities, embedding_dim)
        self.ent_im = nn.Embedding(num_entities, embedding_dim)

        self.rel_re = nn.Embedding(num_relations, embedding_dim)
        self.rel_im = nn.Embedding(num_relations, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for emb in [
            self.ent_re, self.ent_im,
            self.rel_re, self.rel_im
        ]:
            nn.init.xavier_uniform_(emb.weight)

    def score_triples(self, triples):
        """
        triples: LongTensor of shape (batch_size, 3)
        """
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]

        h_re = self.ent_re(h)
        h_im = self.ent_im(h)
        r_re = self.rel_re(r)
        r_im = self.rel_im(r)
        t_re = self.ent_re(t)
        t_im = self.ent_im(t)

        # ComplEx score
        score = (
            h_re * r_re * t_re
            + h_im * r_re * t_im
            + h_re * r_im * t_im
            - h_im * r_im * t_re
        ).sum(dim=1)

        return score

# %%
def negative_sampling(triples, num_entities):
    """
    Corrupt tails.
    """
    neg_triples = triples.clone()
    neg_triples[:, 2] = torch.randint(
        0, num_entities, (triples.size(0),),
        device=triples.device
    )
    return neg_triples


def train_step(model, optimizer, triples, num_entities):
    model.train()

    # Positive samples
    pos_scores = model.score_triples(triples)

    # Negative samples
    neg_triples = negative_sampling(triples, num_entities)
    neg_scores = model.score_triples(neg_triples)

    # Labels
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    loss = F.binary_cross_entropy_with_logits(scores, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_complex(
    triples,
    num_entities,
    num_relations,
    embedding_dim=200,
    epochs=100,
    lr=1e-3,
    batch_size=1024,
    model=None,
    device="cpu"
):
    if model is None:
        model = ComplEx(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim
        ).to(device)
    else:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    triples = triples.to(device)

    for epoch in range(epochs):
        perm = torch.randperm(triples.size(0))
        total_loss = 0.0

        for i in range(0, triples.size(0), batch_size):
            batch = triples[perm[i:i + batch_size]]
            loss = train_step(model, optimizer, batch, num_entities)
            total_loss += loss

        # if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f}")

    return model

# %%

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE, 'datasets/normalized_base_all_graph.pkl'), 'rb') as fi:
    data = pickle.load(fi)
    full_data = data['data']
    full_index = data['index']

class_index = full_index['class_index']
rev_class_index = {v: k for k,v in class_index.items()}
prop_index = full_index['property_index']
rev_prop_index = {v: k for k,v in prop_index.items()}
# %%
gci0_as_2 = full_data['gci0'].clone()
subclass_index = len(rev_prop_index)
const = subclass_index * torch.ones(len(gci0_as_2), 1, device=gci0_as_2.device, dtype=gci0_as_2.dtype)

print(gci0_as_2.shape)
gci0_as_2 = torch.cat([gci0_as_2[:, :1], const, gci0_as_2[:, 1:]], dim=1)
print(gci0_as_2.shape)
rev_prop_index[subclass_index] = 'subClassOf'
prop_index['subClassOf'] = subclass_index

# %%
triples = torch.cat([full_data['gci2'], gci0_as_2], dim=0)
# %%
complex_model = train_complex(
    triples=triples,
    num_entities=len(rev_class_index),
    num_relations=len(rev_prop_index),
    embedding_dim=32,
    epochs=200,
    lr=1e-3,
    batch_size=2**17,
    # model=complex_model,
    device="cpu"
)
# %%
with open('complex32.pkl', 'wb') as fo:
    pickle.dump({'re': complex_model.ent_re, 'im': complex_model.ent_im}, fo)
# %%
