library(phyloseq)
library(phytools)
library(ape)
library(dplyr)
library(tidyr)
library(stringi)
library(stringr)


load("examples/datasetA/OSA_tree_filtered.RData")
load("examples/datasetA/OSA_phylo_filtered.RData")

plot(ladderize(osa_tree_new), show.tip.label = F, use.edge.length = T,show.node.label=FALSE,
     direction="downwards", cex = 0.8, type = "fan", label.offset = 0.05)
otu_full <- osa_tree_new$tip.label

osa <- read.csv("examples/datasetA/Microbiome_OSA_data_processed.csv")
osa_air <- read.csv("examples/datasetA/OSA_AIR_top50.csv")
osa_ihh <- read.csv("examples/datasetA/OSA_IHH_top50.csv")

osa_air <- osa_air %>% subset(X != "X0")
osa_ihh <- osa_ihh %>% subset(X != "XO")
rownames(osa) <- osa$X.OTU.ID
rownames(osa_air) <- osa_air$X
rownames(osa_ihh) <- osa_ihh$X
otu_air <- osa %>% subset(rownames(osa) %in% rownames(osa_air))
otu_ihh <- osa %>% subset(rownames(osa) %in% rownames(osa_ihh))

otu_0 <- otu_air$ID
otu_1 <- otu_ihh$ID

edge_full <- osa_tree_new$edge

osa_full_new <- osa_tree_new
otu_sub_0_in_otu_full <- otu_full %in% otu_0
otu_sub_1_in_otu_full <- otu_full %in% otu_1

otu_full_0 <- rep("", length(otu_full))
otu_full_0[otu_sub_0_in_otu_full] <- rep("+", sum(otu_sub_0_in_otu_full))
otu_full_0[otu_sub_1_in_otu_full] <- rep("*", sum(otu_sub_1_in_otu_full))

osa_full_new$tip.label <- otu_full_0

# IHH cases
tipcol <- rep("black", length(osa_full_new$tip.label))
for (i in 1:length(osa_full_new$tip.label)) {
  if (osa_full_new$tip.label[i] == "*") {
    tipcol[i] <- "red"
  } 
}
plot(ladderize(osa_full_new), type =  "fan", label.offset = 0.05,
     show.tip.label = T, cex = 2, tip.color = tipcol)
add.scale.bar()

# Controls group
tipcol <- rep("black", length(osa_full_new$tip.label))
for (i in 1:length(osa_full_new$tip.label)) {
  if (osa_full_new$tip.label[i] == "+") {
    tipcol[i] <- "purple"
  } 
}
plot(ladderize(osa_full_new), type =  "fan", label.offset = 0.05,
     show.tip.label = T, cex = 2, tip.color = tipcol)
add.scale.bar()
